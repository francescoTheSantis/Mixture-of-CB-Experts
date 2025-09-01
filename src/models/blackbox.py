import torch
import torch.nn as nn
from src.models.base import BaseModel
from src.models.encoders.base import BaseEncoder
from src.models.encoders.mlp import MLPEncoder

class BlackBox(BaseModel):
    def __init__(self,
                 output_size,
                 c_names,
                 y_names,
                 task,
                 task_penalty=0.1,
                 hard_concepts=False,
                 activation='ReLU',
                 int_prob=0.1,
                 int_idxs=None,
                 noise=None,
                 latent_size=64,
                 c_groups=None,
                 encoder: BaseEncoder=None,
                 backbone_latent_size=None,
                 concept_type='binary' 
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
                 concept_type
                 )

        self.has_concepts = False
        hidden_size = latent_size 
        
        # self.predictor = nn.Sequential(
        #     nn.Linear(backbone_latent_size, hidden_size),
        #     getattr(nn, activation)(),
        #     nn.Linear(hidden_size, output_size)
        # )

        self.predictor = MLPEncoder(
            backbone_latent_size,
            output_size,
            None,
            hidden_size,
            activation
        )


    def forward(self, input):
        x = input['x']
        x = self.encoder(x)
        y_hat = self.predictor(x)
        return y_hat, None
    
    def loss(self, y_hat, y, c_hat=None, c=None):
        if self.task == 'classification' and self.output_size > 1:
            y = y.flatten().long()
        elif self.task == 'generation':
            y = y.flatten().long()
        elif self.output_size == 1:
            y = y.flatten().float()
        else:
            raise NotImplementedError(f"Task {self.task} is not implemented. "
                                      f"Supported tasks are 'classification', "
                                      f"'regression', and 'generation'.")
        # cross entropy
        loss = self.task_loss_form(y_hat.squeeze(), y)
        return loss