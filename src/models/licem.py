import torch
import torch.nn as nn
import torch_concepts.nn as pyc_nn
from src.models.base import BaseModel
from torch_concepts.nn import functional as CF

class LinearConceptEmbeddingModel(BaseModel):
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
                 use_bias=True,
                 weight_reg=1e-4,
                 bias_reg=1e-4,
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
        self.use_bias = use_bias
        self.y_names = list(y_names)
        self.hard_concepts = hard_concepts

        self.bottleneck = pyc_nn.ConceptEmbeddingBottleneck(
            latent_size,
            self.c_names,
            embedding_size,
        )
        # module predicting the concept importance for all concepts and tasks
        # input batch_size x concept_number x embedding_size
        # output batch_size x concept_number x task_number
        self.concept_relevance = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, latent_size),
            getattr(nn, activation)(),
            torch.nn.Linear(latent_size, len(self.y_names)),
            pyc_nn.Annotate([self.c_names, self.y_names], [1, 2])
        )
        # module predicting the class bias for each class
        # input batch_size x concept_number x embedding_size
        # output batch_size x task_number
        if self.use_bias:
            self.bias_predictor = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(
                    len(self.c_names) * embedding_size,
                    embedding_size,
                ),
                getattr(nn, activation)(),
                torch.nn.Linear(embedding_size, len(self.y_names)),
                pyc_nn.Annotate(self.y_names, 1)
            )

        self.weight_reg = weight_reg
        self.bias_reg = bias_reg
        self.__predicted_weights = None
        if self.use_bias:
            self.__predicted_bias = None

        self.concept_loss_form = nn.BCELoss()

    def forward(self, input):
        latent, c_true, int_idxs = self.encode(input)
        
        c_emb, c_dict = self.bottleneck(
            latent,
            c_true=c_true,
            intervention_idxs=int_idxs,
            intervention_rate=1.,
        )
        c_pred = c_dict['c_int']
        c_input = (c_pred > 0.5).float() if self.hard_concepts else c_pred

        # adding memory dimension to concept weights
        c_weights = self.concept_relevance(c_emb).unsqueeze(dim=1)
        self.__predicted_weights = c_weights

        y_bias = None
        if self.use_bias:
            # adding memory dimension to bias
            y_bias = self.bias_predictor(c_emb).unsqueeze(dim=1)
            self.__predicted_bias = y_bias

        y_pred = CF.linear_equation_eval(c_weights, c_input, y_bias)
        return y_pred[:, :, 0], c_pred
    
    def loss(self, y_hat, y, c_hat=None, c=None):
        loss = self.concept_based_loss(y_hat, y, c_hat, c)
        # adding l1 regularization to the weights
        w_loss = self.weight_reg * self.__predicted_weights.norm(p=2)
        loss += w_loss
        # adding l2 regularization to the biases if used
        if self.use_bias:
            b_loss = self.bias_reg * self.__predicted_bias.norm(p=1)
            loss += b_loss
        return loss


