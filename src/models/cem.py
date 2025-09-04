import torch
import torch.nn as nn
import torch_concepts.nn as pyc_nn
from torch_concepts.nn import concept_embedding_mixture
from src.models.encoders.mlp import MLPEncoder

from src.models.base import BaseModel

class ConceptEmbeddingModel(BaseModel):
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
                 embedding_size = 16,
                 latent_size = 128,
                 c_groups=None,
                 hard_concepts=False,
                 encoder=None,
                 concept_loss_form=nn.BCELoss(),
                 backbone_latent_size=None,
                 concept_type='binary',
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
                 concept_type
                 )

        self.embedding_size = embedding_size
        self.has_concepts = True

        self.bottleneck = pyc_nn.ConceptEmbeddingBottleneck(
            backbone_latent_size,
            self.c_names,
            embedding_size,
            nn.Identity()
        )

        self.y_predictor = MLPEncoder(
            len(self.c_names) * embedding_size,
            output_size,
            None,
            latent_size,
            activation
        )


    def forward(self, input):
        x, c_true, int_idxs = self.encode(input)
        
        c_emb, c_dict = self.bottleneck(
            x,
            c_true=c_true,
            intervention_idxs=int_idxs,
            intervention_rate=1.,
        )
        c_hat = c_dict['c_int']

        c_hat, input_concepts = self._process_concepts(c_hat, c_true, int_idxs)

        # It is necessary to compute again since 
        c_emb = self.bottleneck.linear(x)
        c_emb = concept_embedding_mixture(c_emb, input_concepts)

        y_hat = self.y_predictor(c_emb.flatten(-2))
        return {
            'y_hat': y_hat,
            'c_hat': c_hat
        }

    def loss(self, y_hat, y, c_hat=None, c=None):
        loss = self.concept_based_loss(y_hat, y, c_hat, c)
        return loss


