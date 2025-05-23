from typing import Optional
from torch import nn
import torch
import pytorch_lightning as pl
from src.metrics import Task_Accuracy, Concept_Accuracy
from collections import OrderedDict

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
                ):
        super(Engine, self).__init__()         
        self.model = model
        self.save_hyperparameters(ignore=["model"], logger=False)

        self.c_names = c_names
        self.y_name = y_name

        self.task_metric = Task_Accuracy()
        self.concept_metric = Concept_Accuracy()

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
        y_loss, c_loss = self.model.filter_output_for_loss(*model_output)
        loss = self.model.loss(y_loss, y, c_loss, c)
        return loss, model_output, y, c

    def training_step(self, batch, batch_idx):
        self.model.current_epoch = self.current_epoch
        loss, model_output, y, c = self.shared_step(batch)
        self.log("train_loss", loss)
        output_x_metrics = self.model.filter_output_for_metric(*model_output)
        task_acc = self.task_metric(output_x_metrics[0], y)
        self.log('train_task_acc', task_acc)
        if self.model.has_concepts:
            concept_acc = self.concept_metric(output_x_metrics[1], c)
            self.log('train_concept_acc', concept_acc)
        # If the name of the class is LinearMemoryReasoner,
        # compute the selection entropy
        if self.model.__class__.__name__ == 'LinearMemoryReasoner':
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
        y_output, c_output = self.model.filter_output_for_metric(*model_output)
        task_acc = self.task_metric(y_output, y)
        self.log('val_task_acc', task_acc)
        if self.model.has_concepts:
            concept_acc = self.concept_metric(c_output, c)
            self.log('val_concept_acc', concept_acc)
        # If the name of the class is LinearMemoryReasoner,
        # compute the selection entropy
        if self.model.__class__.__name__ == 'LinearMemoryReasoner':
            # Compute the entropy of the selection distribution
            selection_dist = model_output[3]
            selection_dist = torch.softmax(selection_dist, dim=-1)
            selection_entropy = -torch.sum(selection_dist * torch.log(selection_dist + 1e-10), dim=1)
            selection_entropy = selection_entropy.mean()
            self.log('val_selection_entropy', selection_entropy)
        return loss 
    
    def test_step(self, batch, batch_idx):
        loss, model_output, y, c = self.shared_step(batch)
        y_output, c_output = self.model.filter_output_for_metric(*model_output)
        self.log("test_loss", loss)
        task_acc = self.task_metric(y_output, y)
        self.log('test_task_acc', task_acc)
        if self.model.has_concepts:
            concept_acc = self.concept_metric(c_output, c)
            self.log('test_concept_acc', concept_acc)
        return loss 

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
 