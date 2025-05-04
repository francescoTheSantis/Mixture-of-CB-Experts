import torch
import torch.nn as nn
import torch_concepts.nn as pyc_nn
from src.models.base import BaseModel

class ConceptBottleneckModel(BaseModel):
    def __init__(self, 
                 input_size, 
                 output_size,
                 c_names,
                 task, 
                 task_penalty,
                 task_interpretable=True,
                 activation='ReLU',
                 int_prob=0.1,
                 int_idxs=None,
                 noise=None,
                 latent_size = 128,
                 dataset=None
                 ):
        
        super().__init__(
                 input_size, 
                 output_size,
                 task,
                 activation,
                 latent_size,
                 dataset
                 )

        self.task_interpretable = task_interpretable
        self.task_penalty = task_penalty
        self.c_names = list(c_names)
        self.int_prob = int_prob
        self.int_idxs = int_idxs
        self.has_concepts = True
        self.noise = noise

        self.bottleneck = pyc_nn.LinearConceptBottleneck(
            self.latent_size,
            self.c_names,
        )

        if self.task_interpretable:
            self.y_predictor = nn.Sequential(
                nn.Linear(len(c_names), output_size)
            )
        else:
            self.y_predictor = nn.Sequential(
                nn.Linear(len(c_names), 2 * len(c_names)),
                getattr(nn, activation)(),
                nn.Linear(2 * len(c_names), output_size),
            )

        self.concept_loss_form = nn.BCELoss()

    def forward(self, input):
        x = input['x']
        c_true = input['c']
    
        # If noise is provided, create a convex combination of the input and noise
        if self.noise!=None:
            eps = torch.randn_like(x)
            x = eps * self.noise + x * (1-self.noise)
            
        x = self.encoder(x)

        # If the intervention index is not provided, 
        # all concept can be selected for interventions
        int_idxs = self.int_idxs if self.int_idxs is not None \
            else torch.ones_like(c_true).bool()
        
        c_pred, _ = self.bottleneck(
            x,
            c_true=c_true,
            intervention_idxs=int_idxs,
            intervention_rate=self.int_prob,
        )
        y_pred = self.y_predictor(c_pred)
        return y_pred, c_pred
    
    def filter_output_for_loss(self, y_output, c_output=None):
        return y_output, c_output
    
    def loss(self, y_hat, y, c_hat=None, c=None):
        y = y.flatten().long()
        # task loss
        task_loss = self.task_loss_form(y_hat.squeeze(), y)
        # concept loss
        concept_loss = 0
        for i in range(c.shape[1]):
            concept_loss += self.concept_loss_form(c_hat[:,i], c[:,i])
        # combine the two losses
        loss = concept_loss + self.task_penalty * task_loss
        return loss


