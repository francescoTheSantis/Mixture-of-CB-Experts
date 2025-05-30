import torch
import torch.nn as nn
import torch_concepts.nn as pyc_nn
from torch_concepts.nn import concept_embedding_mixture

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
                 ):

        super().__init__(
                 output_size,
                 task,
                 activation,
                 latent_size,
                 c_groups,
                 encoder
                 )

        self.embedding_size = embedding_size
        self.task_penalty = task_penalty
        self.c_names = list(c_names)
        self.int_prob = int_prob
        self.int_idxs = int_idxs
        self.has_concepts = True
        self.noise = noise
        self.hard_concepts = hard_concepts

        self.bottleneck = pyc_nn.ConceptEmbeddingBottleneck(
            latent_size,
            self.c_names,
            embedding_size,
        )
        self.y_predictor = nn.Sequential(
            nn.Linear(len(self.c_names) * embedding_size, latent_size),
            nn.LeakyReLU(),
            nn.Linear(latent_size, output_size),
        )

        self.concept_loss_form = nn.BCELoss()

    def forward(self, input):
        x, c_true, int_idxs = self.encode(input)
        
        c_emb, c_dict = self.bottleneck(
            x,
            c_true=c_true,
            intervention_idxs=int_idxs,
            intervention_rate=1.,
        )
        c_pred = c_dict['c_int']
        if self.hard_concepts:
            c_emb = self.bottleneck.linear(x)
            c_pred_hard = (c_pred > 0.5).float()
            c_emb = concept_embedding_mixture(c_emb, c_pred_hard)

        y_pred = self.y_predictor(c_emb.flatten(-2))
        return y_pred, c_pred

    def loss(self, y_hat, y, c_hat=None, c=None):
        loss = self.concept_based_loss(y_hat, y, c_hat, c)
        return loss


