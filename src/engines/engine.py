from typing import Optional
from torch import nn
import torch
import pytorch_lightning as pl
from src.metrics import MAE, MSE, ClassAccuracy
from collections import OrderedDict
import pandas as pd
import torch.nn.functional as F
from torchmetrics import Metric, MetricCollection
import sympy as sp

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
                data_path: Optional[str] = None,
                scale_target: bool = True
                ):
        super(Engine, self).__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"], logger=False)
        self.data_type = data_type
        self.c_names = c_names
        self.y_name = y_name
        self.num_classes = len(y_name) #if len(y_name)>1 else 2
        self.class_names = y_name # if len(y_name)>1 else ['0','1']
        self.model_name = self.model.__class__.__name__

        self.csv_log_dir = csv_log_dir
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.scale_target = scale_target

        # Set the metrics
        self._set_metrics()

        # If we are using the LinearMemoryReasoner model,
        # we need to save the tensors required for the explanations.
        if self.model_name in ['LinearMemoryReasoner', 'SymbolicMemoryReasoner']:
            self.explanations = []
            self.c_trues = []
            self.c_preds = []
            self.y_trues = []
            self.y_preds = []

    @staticmethod
    def _check_metric(metric):
        metric = metric.clone()
        metric.reset()
        return metric

    def _set_metrics(self):
        if self.model.task == 'classification':
            self.train_y_metrics = MetricCollection(metrics={'acc': self._check_metric(ClassAccuracy())}, prefix="train/y/")
            self.val_y_metrics = MetricCollection(metrics={'acc': self._check_metric(ClassAccuracy())}, prefix="val/y/")
            self.test_y_metrics = MetricCollection(metrics={'acc': self._check_metric(ClassAccuracy())}, prefix="test/y/")
            self.train_c_metrics = MetricCollection(metrics={'acc': self._check_metric(ClassAccuracy())}, prefix="train/c/")
            self.val_c_metrics = MetricCollection(metrics={'acc': self._check_metric(ClassAccuracy())}, prefix="val/c/")
            self.test_c_metrics = MetricCollection(metrics={'acc': self._check_metric(ClassAccuracy())}, prefix="test/c/")
        elif self.model.task == 'regression':
            self.train_y_metrics = MetricCollection(metrics={'mse': self._check_metric(MSE()),
                                                             'mae': self._check_metric(MAE())}, prefix="train/y/")
            self.val_y_metrics = MetricCollection(metrics={'mse': self._check_metric(MSE()),
                                                           'mae': self._check_metric(MAE())}, prefix="val/y/")
            self.test_y_metrics = MetricCollection(metrics={'mse': self._check_metric(MSE()),
                                                            'mae': self._check_metric(MAE())}, prefix="test/y/")
            self.train_c_metrics = MetricCollection(metrics={'mse': self._check_metric(MSE()),
                                                             'mae': self._check_metric(MAE())}, prefix="train/c/")
            self.val_c_metrics = MetricCollection(metrics={'mse': self._check_metric(MSE()),
                                                           'mae': self._check_metric(MAE())}, prefix="val/c/")
            self.test_c_metrics = MetricCollection(metrics={'mse': self._check_metric(MSE()),
                                                            'mae': self._check_metric(MAE())}, prefix="test/c/")
        else:
            raise NotImplementedError(f"Metrics for task={self.model.task} not implemented.")
        
    def update_and_log_metrics(self, stage, y_hat, y, c_hat, c):
        # update and log task metrics
        y_collection = getattr(self, f"{stage}_y_metrics")
        y_collection.update(y_hat, y)
        self.log_dict(y_collection, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        # update and log concept metrics
        if self.model.has_concepts:
            c_collection = getattr(self, f"{stage}_c_metrics")
            c_collection.update(c_hat, c)
            self.log_dict(c_collection, on_step=False, on_epoch=True, logger=True, prog_bar=True)

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
        # batch['x'] will be a tensor for image and toy datasets, and a dict for text datasets.

        # model forward
        model_output = self.forward(batch)

        # Compute loss
        y_hat_loss, c_hat_loss = self.model.filter_output_for_loss(**model_output)
        if self.model.task == 'regression' and self.scale_target:
            y_loss = self.scaler.transform(batch['y'])
        else:
            y_loss = batch['y']
        loss = self.model.loss(y_hat_loss, y_loss, c_hat_loss, batch['c'])
        
        # return everything
        return loss, model_output

    def training_step(self, batch, batch_idx):
        self.model.global_step = self.global_step
        loss, model_output = self.shared_step(batch)
        self.log("train_loss", loss.item())

        y_hat_metrics, c_hat_metrics = self.model.filter_output_for_metrics(**model_output)
        # compute task metrics
        # if the task is regression, we denormalize the target variable to compute the metrics
        if self.model.task == 'regression' and self.scale_target:
            y_hat_metrics = self.scaler.inverse_transform(y_hat_metrics.detach())
        self.update_and_log_metrics('train', y_hat_metrics, batch['y'], c_hat_metrics, batch['c'])

        # compute the selection entropy
        if self.model_name in ['LinearMemoryReasoner', 'SymbolicMemoryReasoner']:
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
        if self.model.__class__.__name__ == 'SymbolicMemoryReasoner':
            if self.model.equation_learning_strategy == 'kan':
                # Prune the KAN layers
                for kan_layer in self.model.kan_layers:
                    # prune each kan layer
                    kan_layer.prune()

                    # Set the symbolic equation in the kan layer
                    kan_layer.auto_symbolic()

                # Fine-tune the model after replacing the KAN layers with the symbolic equations.
                # TODO ...

    def validation_step(self, batch, batch_idx):
        loss, model_output = self.shared_step(batch)
        self.log("val_loss", loss.item())

        y_hat_metrics, c_hat_metrics = self.model.filter_output_for_metrics(**model_output)
        # compute task metrics
        # if the task is regression, we denormalize the target variable to compute the metrics
        if self.model.task == 'regression' and self.scale_target:
            y_hat_metrics = self.scaler.inverse_transform(y_hat_metrics)
        self.update_and_log_metrics('val', y_hat_metrics, batch['y'], c_hat_metrics, batch['c'])

        # compute the selection entropy
        if self.model_name in ['LinearMemoryReasoner', 'SymbolicMemoryReasoner']:
            # Compute the entropy of the selection distribution
            selection_dist = model_output['selection_dist']
            selection_dist = torch.softmax(selection_dist, dim=-1)
            selection_entropy = -torch.sum(selection_dist * torch.log(selection_dist + 1e-10), dim=1)
            selection_entropy = selection_entropy.mean()
            self.log('val_selection_entropy', selection_entropy)
        return loss 
    
    def test_step(self, batch, batch_idx):
        loss, model_output = self.shared_step(batch)
        self.log("test_loss", loss.item())

        y_hat_metrics, c_hat_metrics = self.model.filter_output_for_metrics(**model_output)
        # compute task metrics
        # if the task is regression, we denormalize the target variable to compute the metrics
        if self.model.task == 'regression' and self.scale_target:
            y_hat_metrics = self.scaler.inverse_transform(y_hat_metrics)
        self.update_and_log_metrics('test', y_hat_metrics, batch['y'], c_hat_metrics, batch['c'])

        # update the tensors required for the explanations.
        c = batch['c']
        y = batch['y']
        if self.model.__class__.__name__ in ['LinearMemoryReasoner', 'SymbolicMemoryReasoner']:
            if self.model.__class__.__name__ == 'LinearMemoryReasoner':
                self.explanations.append(model_output['explanations'])
            else:
                for eq in model_output['explanations']:
                    self.explanations.append(eq)
            self.c_trues.append(c)
            self.c_preds.append(c_hat_metrics)
            self.y_trues.append(y)
            self.y_preds.append(y_hat_metrics)
        return loss 
    
    def _denormalize(self, tensor):
        if self.model.task == 'regression':
            return tensor * self.model.y_std + self.model.y_mean
        else:
            return tensor


    def on_test_epoch_end(self):

        model_name = self.model.__class__.__name__

        # The whole function is only executed for: LinearMemoryReasoner, SymbolicMemoryReasoner.
        if model_name in ['LinearMemoryReasoner', 'SymbolicMemoryReasoner']:
            # If the name of the class is LinearMemoryReasoner,
            # store the tensors required for the explanations.
            if model_name == 'LinearMemoryReasoner':
                # Concatenate the tensors
                self.explanations = torch.cat(self.explanations, dim=0)
                # Save the predicted_CBM to a .pt file
                torch.save(self.explanations, f"{self.csv_log_dir}/pred_CBMs.pt")
            elif model_name == 'SymbolicMemoryReasoner':
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