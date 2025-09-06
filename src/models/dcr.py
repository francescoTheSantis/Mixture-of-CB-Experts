import torch
import torch.nn as nn
import torch_concepts.nn as pyc_nn
from torch_concepts.semantic import ProductTNorm
from torch_concepts.nn import functional as CF
from src.models.base import BaseModel
from torch.nn import functional as F
from torch_concepts.nn import concept_embedding_mixture

class DeepConceptReasoner(BaseModel):
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
                 semantic = ProductTNorm(),
                 temperature = 100,
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
        self.logic_reasoning = True
        self.n_roles = 3
        self.memory_names = ['Positive', 'Negative', 'Irrelevant']
        
        self.embedding_size = embedding_size
        self.task_penalty = task_penalty * 3 # BCE gives lower loss values
        self.has_concepts = True
        self.semantic = semantic
        self.temperature = temperature

        self.bottleneck = pyc_nn.ConceptEmbeddingBottleneck(
            backbone_latent_size,
            self.c_names,
            embedding_size,
            activation=nn.Identity()
        )
        self.concept_importance_predictor = nn.Sequential(
            nn.Linear(embedding_size, self.latent_size),
            getattr(nn, activation)(),
            nn.Linear(self.latent_size, output_size * self.n_roles),
            nn.Unflatten(-1, (output_size, self.n_roles)),
        )

        self.task_loss_form = nn.BCELoss()

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

        c_weights = self.concept_importance_predictor(c_emb)
        # adding memory dimension
        c_weights = c_weights.unsqueeze(dim=1)
        # soft selecting concept relevance (last role) among concepts
        relevance = CF.soft_select(c_weights[:, :, :, :, -2:-1],
                                   self.temperature, -3)
        # softmax over positive/negative roles
        polarity = c_weights[:, :, :, :, :-1].softmax(-1)
        # batch_size x memory_size x n_concepts x n_tasks x n_roles
        c_weights = torch.cat([polarity, 1 - relevance], dim=-1)

        y_hat = CF.logic_rule_eval(c_weights, input_concepts,
                                    semantic=self.semantic)
        # removing memory dimension
        y_hat = y_hat[:, :, 0]
        return {
            'y_hat': y_hat,
            'c_hat': c_hat
        }

    def loss(self, y_hat, y, c_hat=None, c=None):
        loss = self.concept_based_loss(y_hat, y, c_hat, c)
        return loss