from typing import Optional
from torch import nn
import torch
import pytorch_lightning as pl
from src.metrics import Task_Accuracy, Concept_Accuracy
from collections import OrderedDict

class Engine(pl.LightningModule):    
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
        y_output, c_output = self.model.filter_output_for_loss(*model_output)
        loss = self.model.loss(y_output, y, c_output, c)
        return loss, y_output, c_output, y, c

    def training_step(self, batch, batch_idx):
        self.model.current_epoch = self.current_epoch
        loss, _, _, _, _ = self.shared_step(batch)
        self.log("train_loss", loss)
        return loss      

    def validation_step(self, batch, batch_idx):
        loss, y_output, c_output, y, c = self.shared_step(batch)
        self.log("val_loss", loss)
        task_acc = self.task_metric(y_output, y)
        self.log('val_task_acc', task_acc)
        if self.model.has_concepts:
            concept_acc = self.concept_metric(c_output, c)
            self.log('val_concept_acc', concept_acc)
        return loss 
    
    def test_step(self, batch, batch_idx):
        loss, y_output, c_output, y, c = self.shared_step(batch)
        self.log("test_loss", loss)
        task_acc = self.task_metric(y_output, y)
        self.log('test_task_acc', task_acc)
        if self.model.has_concepts:
            concept_acc = self.concept_metric(c_output, c)
            self.log('test_concept_acc', concept_acc)
        return loss 

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
 