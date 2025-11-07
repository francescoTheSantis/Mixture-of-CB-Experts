import torch
import torch.nn as nn
from src.models.baselines.base import BaseModel
from src.models.encoders.base import BaseEncoder
from src.models.encoders.mlp import MLPEncoder
from src.utils.expression_utils import store_eq

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
                 concept_type='binary',
                 disjoint_training=False,
                 num_layers=1,
                 concept_penalty=1.0,
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
                 disjoint_training,
                 concept_penalty
                 )

        self.has_concepts = False
        hidden_size = latent_size 

        self.predictor = MLPEncoder(
            input_size=backbone_latent_size,
            output_size=output_size,
            input_transform=None,
            hidden_size=hidden_size,
            activation=activation,
            num_layers=num_layers,
        )

    def forward(self, input):
        x = input['x']
        x = self.encoder(x)
        y_hat = self.predictor(x)
        return {
            'y_hat': y_hat,
            'c_hat': None
        }

    def loss(self, y_hat, y, *args, **kwargs):
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
    
    def get_symbolic_equivalent(self, log_dir=None):
        """
        Returns the equation associated to the predictor of the model
        """

        # Get as many equations as the output size
        equations = self.predictor.to_symbolic()

        # If the output is greater than 1, equations will be a list.
        # Each equation in the list will have the same complexity, therefore we return only the first one.
        if self.output_size > 1:
            store_eq(equations[0], log_dir)
        store_eq(equations, log_dir)


        

        
        