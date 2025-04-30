import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.metrics import Task_Accuracy, Concept_Accuracy
from src.utilities import get_intervened_concepts_predictions
from torchvision.models import resnet34

class ConceptBottleneckModel(pl.LightningModule):
    def __init__(self, 
                 input_dim, 
                 n_concepts, 
                 n_labels,
                 task_penalty,
                 task_interpretable=True,
                 p_int_train=None,
                 train_backbone=False):
        super().__init__()

        # Encoder: Maps input to concepts
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, n_concepts),
            nn.Sigmoid()
        )

        # Decoder: Maps concepts to output
        if task_interpretable:
            self.decoder = nn.Sequential(
                nn.Linear(n_concepts, n_labels)
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(n_concepts, n_concepts),
                nn.ReLU(),
                nn.Linear(n_concepts, n_labels)
            )

        self.has_concepts = True
        self.task_metric = Task_Accuracy()
        self.concept_metric = Concept_Accuracy()
        self.task_penalty = task_penalty
        self.p_int_train = p_int_train

        self.train_backbone = train_backbone
        if self.train_backbone:
            self.setup_backbone()

    def setup_backbone(self):
        self.backbone = resnet34(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        # freeze all the layers except the last one
        for param in self.backbone.parameters():
            param.requires_grad = True
        # Unfreeze the last layer
        #for param in self.backbone[-2].parameters():
        #    param.requires_grad = True
        #for param in self.backbone[-1].parameters():
        #    param.requires_grad = True
        print('Backbone setup done!')

    def forward(self, x, concept_labels, noise=None, p_int=None):
        p_int = self.p_int_train if self.training else p_int
        if self.train_backbone:
            x = self.backbone(x)
            x = x.flatten(start_dim=1)
        if noise!=None:
            eps = torch.randn_like(x)
            x = eps * noise + x * (1-noise)
        concepts = self.encoder(x)
        if p_int!=None:
            concepts = get_intervened_concepts_predictions(concepts, concept_labels, p_int, False, self.training)
        output = self.decoder(concepts)
        return concepts, output

    def step(self, batch, batch_idx, noise=None, p_int=None):
        x, c, y = batch
        c_pred, y_hat = self.forward(x, c, noise, p_int)
        task_loss = F.cross_entropy(y_hat, y.long())
        concept_loss = 0
        for i in range(c.shape[1]):
            concept_loss += F.binary_cross_entropy(c_pred[:,i], c[:,i])
        concept_loss /= c.shape[1]
        loss = concept_loss + task_loss * self.task_penalty
        return loss, task_loss, concept_loss, c, y, c_pred, y_hat

    def training_step(self, batch, batch_idx):
        loss, task_loss, concept_loss, _, _, _, _ = self.step(batch, batch_idx)
        self.log('train_task_loss', task_loss*self.task_penalty)
        self.log('train_concept_loss', concept_loss)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, task_loss, concept_loss, _, _, _, _ = self.step(batch, batch_idx)
        self.log('val_task_loss', task_loss*self.task_penalty)
        self.log('val_concept_loss', concept_loss)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, task_loss, concept_loss, c, y, c_pred, y_hat = self.step(batch, batch_idx)

        task_acc = self.task_metric(y_hat, y)
        self.log('test_task_acc', task_acc)

        concept_acc = self.concept_metric(c_pred, c)
        self.log('test_concept_acc', concept_acc)
        return loss
    
    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

