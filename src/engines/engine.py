from typing import Optional
from torch import nn
import torch
import pytorch_lightning as pl
from src.metrics import Task_Accuracy, Concept_Accuracy
from collections import OrderedDict
import pandas as pd
import torch.nn.functional as F

class Engine(pl.LightningModule):  
    """
    PyTorch Lightning module wrapper.

    Args:
        model (Optional[nn.Module]): The pytorch model to train.
        c_names (Optional[list]): List of concept names.
        y_name (Optional[str]): Target variable name.

    Attributes:
        model (nn.Module): The wrapped model.
        c_names (list): List of concept names.
        y_name (str): Target variable name.
        task_metric (Task_Accuracy): Metric to evaluate task prediction accuracy.
        concept_metric (Concept_Accuracy): Metric to evaluate concept prediction accuracy.

    Methods:
        forward(input): Forward pass through the model.
        predict(input): Alias for forward.
        unpack_batch(batch): Extracts inputs, concepts, and targets from a batch.
        shared_step(batch): Performs a forward pass, computes loss, and returns outputs and labels.
        training_step(batch, batch_idx): Executes one training step and logs training loss.
        validation_step(batch, batch_idx): Executes one validation step, computes and logs loss and accuracies.
        test_step(batch, batch_idx): Executes one test step, computes and logs loss and accuracies.
        configure_optimizers(): Returns the optimizer and learning rate scheduler.
    """
    def __init__(self,
                model: Optional[nn.Module] = None,
                c_names: Optional[list] = None,
                y_name: Optional[str] = None,
                csv_log_dir: Optional[str] = None,
                ):
        super(Engine, self).__init__()         
        self.model = model
        self.save_hyperparameters(ignore=["model"], logger=False)

        self.c_names = c_names
        self.y_name = y_name

        self.task_metric = Task_Accuracy()
        self.concept_metric = Concept_Accuracy()

        self.csv_log_dir = csv_log_dir

        # If we are using a the PredictCBM model,
        # we need to save the tensors required for the explanations.
        if self.model.__class__.__name__ == 'PredictCBM':
            self.pred_CBMs = []
            self.c_trues = []
            self.c_preds = []
            self.y_trues = []
            self.y_preds = []

    def forward(self, input):
        return self.model(input)

    def predict(self, input):
        return self.model(input)

    def unpack_batch(self, batch):
        x = batch[0]
        c = batch[1]
        y = batch[2]
        return x, c, y

    def shared_step(self, batch):
        x, c, y = self.unpack_batch(batch)
        inputs = {'x':x, 'c':c, 'y':y.float()}
        # model forward
        model_output = self.forward(inputs)
        # Compute loss
        y_output, c_output = self.model.filter_output_for_loss(*model_output)
        loss = self.model.loss(y_output, y, c_output, c)
        return loss, model_output, y, c

    def training_step(self, batch, batch_idx):
        self.model.current_epoch = self.current_epoch
        loss, model_output, y, c = self.shared_step(batch)
        self.log("train_loss", loss)
        output_x_metrics = self.model.filter_output_for_metrics(*model_output)
        task_acc = self.task_metric(output_x_metrics[0], y) #TODO: check we have an output list for all models
        self.log('train_task_acc', task_acc)
        if self.model.has_concepts:
            concept_acc = self.concept_metric(output_x_metrics[1], c)
            self.log('train_concept_acc', concept_acc)
        # If the name of the class is PredictCBM,
        # compute the selection entropy
        if self.model.__class__.__name__ == 'PredictCBM':
            # Compute the entropy of the selection distribution
            selection_dist = model_output[3]
            selection_dist = torch.softmax(selection_dist, dim=-1)
            selection_entropy = -torch.sum(selection_dist * torch.log(selection_dist + 1e-10), dim=1)
            selection_entropy = selection_entropy.mean()
            self.log('train_selection_entropy', selection_entropy)
        return loss      

    def validation_step(self, batch, batch_idx):
        loss, model_output, y, c = self.shared_step(batch)
        self.log("val_loss", loss)
        y_output, c_output = self.model.filter_output_for_metrics(*model_output)
        task_acc = self.task_metric(y_output, y)
        self.log('val_task_acc', task_acc)
        if self.model.has_concepts:
            concept_acc = self.concept_metric(c_output, c)
            self.log('val_concept_acc', concept_acc)
        # If the name of the class is PredictCBM,
        # compute the selection entropy
        if self.model.__class__.__name__ == 'PredictCBM':
            # Compute the entropy of the selection distribution
            selection_dist = model_output[3]
            selection_dist = torch.softmax(selection_dist, dim=-1)
            selection_entropy = -torch.sum(selection_dist * torch.log(selection_dist + 1e-10), dim=1)
            selection_entropy = selection_entropy.mean()
            self.log('val_selection_entropy', selection_entropy)
        return loss 
    
    def test_step(self, batch, batch_idx):
        loss, model_output, y, c = self.shared_step(batch)
        y_output, c_output = self.model.filter_output_for_metrics(*model_output)
        self.log("test_loss", loss)
        task_acc = self.task_metric(y_output, y)
        self.log('test_task_acc', task_acc)
        if self.model.has_concepts:
            concept_acc = self.concept_metric(c_output, c)
            self.log('test_concept_acc', concept_acc)

        # If the name of the class is PredictCBM,
        # update the tensors required for the explanations.
        if self.model.__class__.__name__ == 'PredictCBM':
            self.pred_CBMs.append(model_output[2])
            self.c_trues.append(c)
            self.c_preds.append(c_output)
            self.y_trues.append(y)
            self.y_preds.append(y_output)
        return loss 
    
    def on_test_epoch_end(self):
        # If the name of the class is PredictCBM,
        # store the tensors required for the explanations.
        if self.model.__class__.__name__ == 'PredictCBM':
            # Concatenate the tensors
            self.pred_CBMs = torch.cat(self.pred_CBMs, dim=0)
            self.c_trues = torch.cat(self.c_trues, dim=0)
            self.c_preds = torch.cat(self.c_preds, dim=0)
            self.y_trues = torch.cat(self.y_trues, dim=0)
            self.y_preds = torch.cat(self.y_preds, dim=0).argmax(-1)

            # Convert the tensors to pandas dfs
            c_preds = pd.DataFrame(self.c_preds.cpu().numpy(), columns=self.c_names)
            c_trues = pd.DataFrame(self.c_trues.cpu().numpy(), columns=self.c_names)

            # Create a list of names for the y_preds and y_trues to create 
            # a pandas containing the list of predicted and true labels
            y_preds = pd.DataFrame(F.one_hot(self.y_preds.cpu(), len(self.y_name)).numpy(), 
                                   columns=self.y_name)
            y_trues = pd.DataFrame(F.one_hot(self.y_trues.cpu(), len(self.y_name)).numpy(), 
                                   columns=self.y_name)

            # Store the pandas dfs
            c_preds.to_csv(f"{self.csv_log_dir}/c_preds.csv", index=False)
            c_trues.to_csv(f"{self.csv_log_dir}/c_trues.csv", index=False)
            y_preds.to_csv(f"{self.csv_log_dir}/y_preds.csv", index=False)
            y_trues.to_csv(f"{self.csv_log_dir}/y_trues.csv", index=False)

            # Save the predicted_CBM to a .pt file
            torch.save(self.pred_CBMs, f"{self.csv_log_dir}/pred_CBMs.pt")

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
 