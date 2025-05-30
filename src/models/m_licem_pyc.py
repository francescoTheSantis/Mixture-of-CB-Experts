import torch
import torch.nn as nn
import torch_concepts.nn as pyc_nn
from torch_concepts.nn import functional as CF
from src.models.base import BaseModel

class LinearMemoryReasoner(BaseModel):
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
                 memory_size=7,
                 negative_concepts=True,
                 hard_concepts=False,
                 weight_reg=1e-4,
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
        self.latent_size = latent_size
        self.task_penalty = task_penalty
        self.c_names = list(c_names)
        self.int_prob = int_prob
        self.int_idxs = int_idxs
        self.has_concepts = True
        self.noise = noise
        self.y_names = list(y_names)
        self.negative_concepts = negative_concepts
        self.hard_concepts = hard_concepts
        self.weight_reg = weight_reg

        self.memory_size = memory_size

        self.bottleneck = pyc_nn.ConceptEmbeddingBottleneck(
            latent_size,
            self.c_names,
            embedding_size,
        )

        self.classifier_selector = nn.Sequential(
            torch.nn.Linear(embedding_size * len(c_names),
                            latent_size),
            pyc_nn.LinearConceptLayer(
                latent_size,
                [self.y_names, memory_size],
            ),
        )
        self.equation_memory = torch.nn.Embedding(
            memory_size,
            latent_size
        )

        self.equation_decoder = pyc_nn.LinearConceptLayer(
            latent_size,
            [
                self.c_names,
                self.y_names,
            ],
        )

        self.concept_loss_form = nn.BCELoss()

    def forward(self, input):
        latent, c_true, int_idxs = self.encode(input)

        c_emb, c_dict = self.bottleneck(
            latent,
            c_true=c_true,
            intervention_idxs=int_idxs,
            intervention_rate=1,
        )
        c_pred = c_dict['c_int']
        classifier_selector_logits = self.classifier_selector(c_emb.flatten(-2))
        prob_per_classifier = torch.softmax(classifier_selector_logits, dim=-1)
        # adding batch dimension to concept memory
        equation_weights = self.equation_decoder(
            self.equation_memory.weight).unsqueeze(dim=0)

        if self.hard_concepts:
            input_concepts = (c_pred > 0.5).float()
        else:
            input_concepts = c_pred
        if self.negative_concepts:
            input_concepts = 2*input_concepts - 1 #TODO: consider converting into convex combination of positive and weights as in CBM
        else:
            input_concepts = input_concepts

        y_per_classifier = CF.linear_equation_eval(equation_weights, 
                                                   input_concepts,
                                                   None)
        y_pred = CF.selection_eval(prob_per_classifier,
                                   y_per_classifier)
        return y_pred, c_pred
    
    def loss(self, y_hat, y, c_hat=None, c=None):
        loss = self.concept_based_loss(y_hat, y, c_hat, c)
        # Add L1 regularization on the weights of the equation memory
        # to encourage sparsity
        loss += self.weight_reg * self.equation_memory.weight.norm(p=1)
        return loss
