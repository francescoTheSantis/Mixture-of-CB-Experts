import torch
import torch.nn as nn
from src.models.base import BaseModel

class BlackBox(BaseModel):
    def __init__(self,
                 input_size,
                 output_size=2,
                 activation='ReLU',
                 task = 'classification',
                 latent_size = 128,
                 dataset = None
                 ):
        super().__init__(
                 input_size, 
                 output_size,
                 task,
                 activation,
                 latent_size,
                 dataset
                 )
                
        self.has_concepts = False
        hidden_size = input_size * 128
        
        self.predictor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            getattr(nn, activation)(),
            nn.Linear(hidden_size, hidden_size),
            getattr(nn, activation)(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, input):
        x = input['x']
        x = self.encoder(x)
        y_hat = self.predictor(x)
        return y_hat, None
    
    def filter_output_for_loss(self, y_output, c_output=None):
        return y_output, c_output
    
    def loss(self, y_hat, y, c_hat=None, c=None):
        y = y.flatten().long()
        # cross entropy
        loss = self.task_loss_form(y_hat.squeeze(), y)
        return loss