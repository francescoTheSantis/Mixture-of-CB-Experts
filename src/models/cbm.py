import torch
import torch.nn as nn
import torch_concepts.nn as pyc_nn
from src.models.encoders.mlp import MLPEncoder
from src.models.base import BaseModel

class ConceptBottleneckModel(BaseModel):
    def __init__(self, 
                 output_size,
                 c_names,
                 y_names,
                 task, 
                 task_penalty,
                 task_interpretable=True,
                 hard_concepts=False,
                 bias=True,
                 activation='ReLU',
                 int_prob=0.1,
                 int_idxs=None,
                 noise=None,
                 latent_size = 128,
                 c_groups=None,
                 encoder=None,
                 backbone_latent_size=None,
                 concept_type='binary',
                 disjoint_training=False,
                 **kwargs
                 ):
        
        super().__init__(
                 output_size,
                 c_names,
                 y_names,
                 task,
                 task_penalty,
                 hard_concepts,
                 activation,
                 int_prob,
                 int_idxs,
                 noise,
                 latent_size,
                 c_groups,
                 encoder,
                 backbone_latent_size,
                 concept_type,
                 disjoint_training
                 )

        self.task_interpretable = task_interpretable
        self.has_concepts = True

        self.bottleneck = pyc_nn.LinearConceptBottleneck(
            backbone_latent_size,
            self.c_names,
            activation=nn.Identity(), # we will later apply a sigmoid if the concept is boolean
        )

        if self.task_interpretable:
            self.y_predictor = (nn.Linear(len(c_names), output_size, bias=bias))
        else:
            self.y_predictor = MLPEncoder(
                len(c_names),
                output_size,
                None,
                2 * len(c_names),
                activation
            )


    def forward(self, input):
        x, c_true, int_idxs = self.encode(input)
        
        c_hat, _ = self.bottleneck(x)

        c_hat, input_concepts = self._process_concepts(c_hat, c_true, int_idxs)

        y_hat = self.y_predictor(input_concepts)

        return {
            'y_hat': y_hat,
            'c_hat': c_hat
        }

    def loss(self, y_hat, y, c_hat=None, c=None):
        loss = self.concept_based_loss(y_hat, y, c_hat, c)
        return loss