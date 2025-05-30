import torch
import torch.nn as nn
import torch_concepts.nn as pyc_nn
from src.models.base import BaseModel

class ConceptBottleneckModel(BaseModel):
    def __init__(self, 
                 output_size,
                 c_names,
                 y_names,
                 task, 
                 task_penalty,
                 task_interpretable=True,
                 neg_concepts=False,
                 hard_concepts=False,
                 bias=True,
                 activation='ReLU',
                 int_prob=0.1,
                 int_idxs=None,
                 noise=None,
                 latent_size = 128,
                 c_groups=None,
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

        self.task_interpretable = task_interpretable
        self.neg_concepts = neg_concepts
        self.hard_concepts = hard_concepts
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
            if self.neg_concepts:
                self.pos_y_predictor = (
                    nn.Linear(len(c_names), output_size, bias=bias))
                self.neg_y_predictor = (
                    nn.Linear(len(c_names), output_size, bias=bias))
            else:
                self.y_predictor = (
                    nn.Linear(len(c_names), output_size, bias=bias))
        else:
            self.y_predictor = nn.Sequential(
                nn.Linear(len(c_names), 2 * len(c_names)),
                getattr(nn, activation)(),
                nn.Linear(2 * len(c_names), output_size),
            )

        self.concept_loss_form = nn.BCELoss()

    def forward(self, input):
        x, c_true, int_idxs = self.encode(input)
        
        c_pred, _ = self.bottleneck(
            x,
            c_true=c_true,
            intervention_idxs=int_idxs,
            intervention_rate=1.,
        )
        if self.hard_concepts:
            input_concepts = (c_pred > 0.5).float()
        else:
            input_concepts = c_pred
        if self.neg_concepts and self.task_interpretable:
            pos_concepts = input_concepts
            neg_concepts = 1 - input_concepts
            y_pred = self.pos_y_predictor(pos_concepts) + self.neg_y_predictor(neg_concepts)
        else:
            y_pred = self.y_predictor(input_concepts)
        return y_pred, c_pred

    def loss(self, y_hat, y, c_hat=None, c=None):
        loss = self.concept_based_loss(y_hat, y, c_hat, c)
        return loss


