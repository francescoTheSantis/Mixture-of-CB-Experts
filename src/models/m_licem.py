import torch
import torch.nn as nn
import torch_concepts.nn as pyc_nn
from torch_concepts.nn import functional as CF
from src.models.base import BaseModel
from torch.nn import functional as F
from torch_concepts.nn import Annotate
import torch_concepts.nn.functional as CF

class LinearMemoryReasoner(BaseModel):
    def __init__(self, 
                 input_size, 
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
                 memory_size=7
                 ):
        super().__init__(
            input_size,
            output_size,
            task,
            activation,
            latent_size,
            c_groups
        )

        self.embedding_size = embedding_size
        self.latent_size = latent_size
        self.task_penalty = task_penalty
        self.c_names = list(c_names)
        self.int_prob = int_prob
        self.int_idxs = int_idxs
        self.has_concepts = True
        self.noise = noise
        self.y_names = list(y_names)

        self.memory_size = memory_size

        self.concept_emb_bottleneck = torch.nn.Sequential(
            torch.nn.Linear(latent_size, len(self.c_names)*embedding_size),
            torch.nn.Unflatten(-1, (len(self.c_names), embedding_size)),
            Annotate(self.c_names, 1),
        )
        self.concept_score_bottleneck = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, 1),
            torch.nn.Flatten(),
            Annotate(self.c_names, 1),
        )
        self.classifier_selector = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(embedding_size//2*len(self.c_names), len(self.y_names)*memory_size),
            torch.nn.Unflatten(-1, (len(self.y_names), memory_size)),
            Annotate(self.y_names, 1),
        )
        self.latent_concept_memory = torch.nn.Embedding(memory_size, latent_size)
        self.concept_memory_decoder = torch.nn.Sequential(
            # the memory decoder maps to the concept space which has also bias
            torch.nn.Linear(latent_size, (len(self.c_names)+1)*len(self.y_names)),
            torch.nn.Unflatten(-1, (len(self.c_names)+1, len(self.y_names))),
            Annotate([self.c_names+["BIAS"], self.y_names], [1, 2]),
        )

        self.concept_loss_form = nn.BCELoss()

    def forward(self, input):
        latent, c_true, int_idxs = self.encode(input)
        c_emb = self.concept_emb_bottleneck(latent)
        c_pred = self.concept_score_bottleneck(c_emb).sigmoid()
        c_intervened = CF.intervene(c_pred, c_true, int_idxs)
        c_mix = CF.concept_embedding_mixture(c_emb, c_intervened)
        classifier_selector_logits = self.classifier_selector(c_mix)
        prob_per_classifier = torch.softmax(classifier_selector_logits, dim=-1)
        memory_weights = self.concept_memory_decoder(self.latent_concept_memory.weight)
        # add batch dimension
        memory_weights = memory_weights.unsqueeze(dim=0)
        concept_weights = memory_weights[:, :, :len(self.c_names)]
        bias = memory_weights[:, :, -1]

        y_per_classifier = CF.linear_equation_eval(concept_weights, c_pred, bias)
        y_pred = CF.selection_eval(prob_per_classifier, y_per_classifier)
            
        return y_pred, c_pred
    
    def filter_output_for_loss(self, y_output, c_output=None):
        return y_output, c_output
    
    def loss(self, y_hat, y, c_hat=None, c=None):
        if self.task == 'classification':
            y = y.flatten().long()
        loss = self.concept_based_loss(y_hat, y, c_hat, c)
        return loss


