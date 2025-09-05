from typing import Optional
from torch import nn
import torch
import pytorch_lightning as pl
from src.metrics import MAE, Task_Accuracy, Concept_Accuracy
from collections import OrderedDict
import pandas as pd
import torch.nn.functional as F

from src.models.base import BaseModel

class Engine(pl.LightningModule):
    """
    PyTorch Lightning module wrapper.
    """
    def __init__(self,
                model: Optional[BaseModel] = None,
                c_names: Optional[list] = None,
                y_name: Optional[str] = None,
                csv_log_dir: Optional[str] = None,
                data_type: Optional[str] = None,
                dataset_name: Optional[str] = None,
                data_path: Optional[str] = None
                ):
        super(Engine, self).__init__()         
        self.model = model
        self.save_hyperparameters(ignore=["model"], logger=False)
        self.data_type = data_type
        self.c_names = c_names
        self.y_name = y_name
        self.num_classes = len(y_name) #if len(y_name)>1 else 2
        self.class_names = y_name # if len(y_name)>1 else ['0','1']

        self.task_metric = Task_Accuracy(logic_reasoning=self.model._logic_model_checker(),
                                         task=self.model.task)
        self.concept_metric = Concept_Accuracy(task=self.model.task)

        if self.model.task == 'regression':
            self.task_MAE = MAE()
            self.concept_MAE = MAE()

        self.csv_log_dir = csv_log_dir
        self.dataset_name = dataset_name
        self.data_path = data_path

        # names of the concept and task metrics
        if self.model.task in ['classification', 'generation']:
            self.task_metric_name = 'task_acc'
            self.concept_metric_name = 'concept_acc'
        elif self.model.task == 'regression':
            self.task_metric_name = 'task_mse'
            self.concept_metric_name = 'concept_mse'

        # If we are using the LinearMemoryReasoner model,
        # we need to save the tensors required for the explanations.
        if self.model.__class__.__name__ in ['LinearMemoryReasoner', 'SymbolicMemoryReasoner']:
            self.explanations = []
            self.c_trues = []
            self.c_preds = []
            self.y_trues = []
            self.y_preds = []

    def forward(self, input):
        return self.model(input)

    def predict(self, input):
        return self.model(input)

    def unpack_batch(self, batch):
        x = batch['x']
        c = batch['c']
        y = batch['y']

        # assert x.isnan().sum() == 0, "Input tensor contains NaN values"
        # assert c.isnan().sum() == 0, "Concept tensor contains NaN values"
        # assert y.isnan().sum() == 0, "Target tensor contains NaN values"

        return x, c, y

    def shared_step(self, batch):
        # batch['x'] will be a tensor for image and toy datasets,
        # and a dict for text datasets.
        inputs = {
            'x': batch['x'],
            'c': batch['c'],
            'y': batch['y'].float()
        }
        # model forward
        model_output = self.forward(inputs)
        # Compute loss
        y_output, c_output = self.model.filter_output_for_loss(**model_output)
        loss = self.model.loss(y_output, inputs['y'], c_output, inputs['c'])
        return loss, model_output, inputs['y'], inputs['c']

    def training_step(self, batch, batch_idx):
        self.model.global_step = self.global_step
        loss, model_output, y, c = self.shared_step(batch)
        self.log("train_loss", loss.item())
        output_x_metrics = self.model.filter_output_for_metrics(**model_output)
        task_acc = self.task_metric(output_x_metrics[0], y) #TODO: check we have an output list for all models
        self.log(f'train_{self.task_metric_name}', task_acc)
        if self.model.has_concepts:
            concept_acc = self.concept_metric(output_x_metrics[1], c)
            self.log(f'train_{self.concept_metric_name}', concept_acc)
        # compute the selection entropy
        if self.model.__class__.__name__ in ['LinearMemoryReasoner', 'SymbolicMemoryReasoner']:
            # Compute the entropy of the selection distribution
            selection_dist = model_output['selection_dist']
            selection_dist = torch.softmax(selection_dist, dim=-1)
            selection_entropy = -torch.sum(selection_dist * torch.log(selection_dist + 1e-10), dim=1)
            selection_entropy = selection_entropy.mean()
            self.log('train_selection_entropy', selection_entropy)
        return loss

    def on_train_end(self):
        # If the model is the symbolic memory reasoner and
        # KANs are used to learn the equations, we need to
        # update the equations at the end of each epoch.
        if self.model.__class__.__name__ == 'SymbolicMemoryReasoner' and self.model.equation_learning_strategy == 'kan':
            # Prune the KAN layers
            for kan_layer in self.model.kan_layers:
                kan_layer.prune()
            # Update the memory by substituting the KANs with
            # their corresponding symbolic equations.
            self.model.setup_kan_equations()

    def validation_step(self, batch, batch_idx):
        loss, model_output, y, c = self.shared_step(batch)
        self.log("val_loss", loss.item())
        y_output, c_output = self.model.filter_output_for_metrics(**model_output)
        task_acc = self.task_metric(y_output, y)
        self.log(f'val_{self.task_metric_name}', task_acc)
        if self.model.has_concepts:
            concept_acc = self.concept_metric(c_output, c)
            self.log(f'val_{self.concept_metric_name}', concept_acc)
        # compute the selection entropy
        if self.model.__class__.__name__ in ['LinearMemoryReasoner', 'SymbolicMemoryReasoner']:
            # Compute the entropy of the selection distribution
            selection_dist = model_output['selection_dist']
            selection_dist = torch.softmax(selection_dist, dim=-1)
            selection_entropy = -torch.sum(selection_dist * torch.log(selection_dist + 1e-10), dim=1)
            selection_entropy = selection_entropy.mean()
            self.log('val_selection_entropy', selection_entropy)
        return loss 
    
    def test_step(self, batch, batch_idx):
        loss, model_output, y, c = self.shared_step(batch)
        y_output, c_output = self.model.filter_output_for_metrics(**model_output)
        self.log("test_loss", loss.item())
        task_acc = self.task_metric(y_output, y)
        self.log(f'test_{self.task_metric_name}', task_acc)

        if self.model.has_concepts:
            concept_acc = self.concept_metric(c_output, c)
            self.log(f'test_{self.concept_metric_name}', concept_acc)

        # log mae if regression
        if self.model.task == 'regression':
            task_mae = self.task_MAE(y_output, y)
            self.log(f'test_task_mae', task_mae)
            if self.model.has_concepts:
                concept_mae = self.concept_MAE(c_output, c)
                self.log(f'test_concept_mae', concept_mae)

        # update the tensors required for the explanations.
        if self.model.__class__.__name__ in ['LinearMemoryReasoner', 'SymbolicMemoryReasoner']:
            if self.model.__class__.__name__ == 'LinearMemoryReasoner':
                self.explanations.append(model_output['explanations'])
            else:
                for eq in model_output['explanations']:
                    self.explanations.append(eq)
            self.c_trues.append(c)
            self.c_preds.append(c_output)
            self.y_trues.append(y)
            self.y_preds.append(y_output)
        return loss 
    
    def on_test_epoch_end(self):
        # If the name of the class is LinearMemoryReasoner,
        # store the tensors required for the explanations.
        if self.model.__class__.__name__ == 'LinearMemoryReasoner':
            # Concatenate the tensors
            self.explanations = torch.cat(self.explanations, dim=0)
            # Save the predicted_CBM to a .pt file
            torch.save(self.explanations, f"{self.csv_log_dir}/pred_CBMs.pt")
        elif self.model.__class__.__name__ == 'SymbolicMemoryReasoner':
            if self.dataset_name == 'mnist_arithmetic':
                # read the file containing the ordered list of rules
                true_eqs = pd.read_csv(f"{self.data_path}/mnist_arithmetic_equations.csv")
                # read all the rules that have been selected
                selected_eqs = pd.DataFrame(self.explanations, columns=['pred_equation'])
                # combine the two in a single dataframe
                combined_eqs = pd.DataFrame()
                combined_eqs['pred_equation'] = selected_eqs['pred_equation']
                combined_eqs['true_equation'] = true_eqs['equation']
                # save the dataframe to a csv file
                combined_eqs.to_csv(f"{self.csv_log_dir}/pred_equations.csv", index=False)

        self.c_trues = torch.cat(self.c_trues, dim=0)
        self.c_preds = torch.cat(self.c_preds, dim=0)
        self.y_trues = torch.cat(self.y_trues, dim=0)
        if self.num_classes > 2:
            # If the number of classes is greater than 1, we need to take the argmax
            self.y_preds = torch.cat(self.y_preds, dim=0).argmax(-1)
        elif self.num_classes == 1 and not isinstance(self.model.task_loss_form, nn.MSELoss):
            # If the number of classes is 1, we just discretize the predictions
            # to get the predicted labels.
            self.y_preds = (torch.cat(self.y_preds, dim=0) > 0.5).long()
        else:
            self.y_preds = (torch.cat(self.y_preds, dim=0)).long()

        # Convert the tensors to pandas dfs
        c_preds = pd.DataFrame(self.c_preds.cpu().numpy(), columns=self.c_names)
        c_trues = pd.DataFrame(self.c_trues.cpu().numpy(), columns=self.c_names)

        # Create a list of names for the y_preds and y_trues to create 
        # a pandas containing the list of predicted and true labels
        if isinstance(self.model.task_loss_form, nn.MSELoss):
            y_preds = pd.DataFrame(self.y_preds.cpu().numpy(), columns=[self.y_name])
            y_trues = pd.DataFrame(self.y_trues.cpu().numpy(), columns=[self.y_name])
        else:
            if self.num_classes == 1:
                y_preds = pd.DataFrame(self.y_preds.long().cpu().numpy(), columns=[self.y_name])
                y_trues = pd.DataFrame(self.y_trues.long().cpu().numpy(), columns=[self.y_name])
            else:
                y_preds = pd.DataFrame(F.one_hot(self.y_preds.cpu(), self.num_classes).squeeze().numpy(),
                                    columns=self.class_names)
                y_trues = pd.DataFrame(F.one_hot(self.y_trues.long().cpu(), self.num_classes).squeeze().numpy(),
                                    columns=self.class_names)

        # Store the pandas dfs
        c_preds.to_csv(f"{self.csv_log_dir}/c_preds.csv", index=False)
        c_trues.to_csv(f"{self.csv_log_dir}/c_trues.csv", index=False)
        y_preds.to_csv(f"{self.csv_log_dir}/y_preds.csv", index=False)
        y_trues.to_csv(f"{self.csv_log_dir}/y_trues.csv", index=False)

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]