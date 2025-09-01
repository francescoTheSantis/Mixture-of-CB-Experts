import torch.nn as nn
import torch
import torch.nn.functional as F
from src.models.encoders.mlp import MLPEncoder
from src.models.encoders.linear import LinearEncoder
from torch_concepts.nn import concept_embedding_mixture

class ConceptBottleneck(nn.Module):
    """
    Layer to be used to predict the concepts given a latent representation.
    """
    def __init__(
            self, 
            input_size, 
            concept_names, 
            bias=True
        ):

        super(ConceptBottleneck, self).__init__()

        self.input_size = input_size
        self.concept_names = concept_names
        self.bias = bias
        self.fc = nn.Linear(input_size, len(concept_names), bias=bias)

    def forward(self, x, c_true, int_idxs):

        # Predict the concepts
        c = self.fc(x)

        # Apply the concept transformations
        c = self._concept_transform(c)

        # Intervene
        c = self._intervene(c, c_true, int_idxs)
        return c

    
class ConceptEmbedder(nn.Module):
    """
    Layer to be used to predict the concepts and concept embeddings 
    given a latent representation.
    """
    def __init__(
            self, 
            input_size, 
            concept_names, 
            embedding_size,
            bias=True
        ):

        super(ConceptBottleneck, self).__init__()

        self.input_size = input_size
        self.concept_names = concept_names
        self.bias = bias
        self.embedding_size = embedding_size
        self.embedder = nn.Linear(
            input_size, 
            len(concept_names) * embedding_size * 2, 
            bias=bias
        )

        self.predictor = nn.Linear(input_size, len(concept_names), bias=bias)

    def forward(self, x, c_true, int_idxs):

        latent = self.embedder(x)

        # Reshape latent
        latent = latent.view(-1, self.num_concepts, self.embedding_size * 2)

        # Predict the concepts
        c = self.predictor(latent) # Shape: (batch_size, num_concepts)

        # Apply the concept transformations
        c = self._concept_transform(c)

        # Intervene
        c_int = self._intervene(c, c_true, int_idxs)

        # Producte concept embeddings
        c_emb = concept_embedding_mixture(c_emb, c_int)

        return {'int':c_int, 'pred':c}, c_emb
        