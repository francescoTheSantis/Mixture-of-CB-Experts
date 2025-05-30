import torch
import torch.nn as nn
import torch_concepts.nn as pyc_nn
from src.models.base import BaseModel

class ConceptResidualModel(BaseModel):
    def __init__(self, 
                 output_size,
                 c_names,
                 y_names,
                 task, 
                 task_penalty,
                 activation='ReLU',
                 int_prob=0.1,
                 int_idxs=None,
                 noise=None,
                 latent_size = 128,
                 residual_size = 10,
                 c_groups=None,
                 hard_concepts=False,
                 encoder=None
                 ):
        
        super().__init__(
                 output_size,
                 task,
                 activation,
                 latent_size,
                 c_groups,
                 encoder
                 )

        self.task_penalty = task_penalty
        self.c_names = list(c_names)
        self.int_prob = int_prob
        self.int_idxs = int_idxs
        self.has_concepts = True
        self.noise = noise
        self.residual_size = residual_size
        self.hard_concepts = hard_concepts

        self.bottleneck = pyc_nn.LinearConceptResidualBottleneck(
            self.latent_size,
            self.c_names,
            self.residual_size,
        )
        self.y_predictor = nn.Sequential(
            nn.Linear(len(c_names) + residual_size, self.latent_size),
            getattr(nn, activation)(),
            nn.Linear(latent_size, output_size),
        )

        self.concept_loss_form = nn.BCELoss()

    def forward(self, input):
        x, c_true, int_idxs = self.encode(input)

        # If the intervention index is not provided, 
        # all the concept are potential candidates for intervention
        int_idxs = self.int_idxs if self.int_idxs is not None \
            else torch.ones_like(c_true).bool()
        
        # TODO: Handle hard concepts
        c_emb, c_dict = self.bottleneck(
            x,
            c_true=c_true,
            intervention_idxs=int_idxs,
            intervention_rate=1.,
        )
        c_pred = c_dict['c_int']
        y_pred = self.y_predictor(c_emb)
        return y_pred, c_pred
    
    def loss(self, y_hat, y, c_hat=None, c=None):
        loss = self.concept_based_loss(y_hat, y, c_hat, c)
        return loss


