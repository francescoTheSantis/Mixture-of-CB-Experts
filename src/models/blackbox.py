import torch
import torch.nn as nn
from src.models.base import BaseModel

class BlackBox(BaseModel):
    def __init__(self,
                 output_size=2,
                 c_names=None,
                 y_names=None,
                 hard_concepts=False,
                 activation='ReLU',
                 task = 'classification',
                 latent_size = 128,
                 c_groups = None,
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

        self.has_concepts = False
        hidden_size = latent_size * 2
        
        self.predictor = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            getattr(nn, activation)(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, input):
        x = input['x']
        x = self.encoder(x)
        y_hat = self.predictor(x)
        return y_hat, None
    
    def loss(self, y_hat, y, c_hat=None, c=None):
        if self.task == 'classification' and self.output_size > 1:
            y = y.flatten().long()
        else:
            y = y.flatten().float()
        # cross entropy
        loss = self.task_loss_form(y_hat.squeeze(), y)
        return loss