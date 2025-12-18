import torch
import torch.nn as nn
import torch_concepts.nn as pyc_nn
from src.models.baselines.base import BaseModel
from torch_concepts.nn import functional as CF
from torch_concepts.nn import concept_embedding_mixture
from src.utils.expression_utils import linear_classifier_expression, store_eq

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
                 concept_loss_form=nn.BCELoss(),
                 backbone_latent_size=None,
                 concept_type='binary',
                 disjoint_training=False,
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

        self.embedding_size = embedding_size
        self.has_concepts = True
        self.use_bias = use_bias
        self.y_names = list(y_names)

        self.bottleneck = pyc_nn.ConceptEmbeddingBottleneck(
            backbone_latent_size,
            self.c_names,
            embedding_size,
            activation=nn.Identity(),
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

    def continuous_mix(self, c_emb, c_pred):
        """
        Create the concept embedding when the concepts are continuous
        """
        # We take only the first embedding produced by the concept embedding layer:
        c_emb = c_emb[:,:,:self.embedding_size] # (batch_size, num_concepts)
        c_emb = c_emb * c_pred[:,:,None]  # (batch_size, num_concepts, embedding_size)
        return c_emb
    
    def forward(self, input):
        latent, _, c_true, int_idxs = self.encode(input)
        
        c_emb, c_dict = self.bottleneck(
            latent,
            c_true=c_true,
            intervention_idxs=int_idxs,
            intervention_rate=1.,
        )
        c_hat = c_dict['c_int']

        c_hat, input_concepts = self._process_concepts(c_hat, c_true, int_idxs)

        # It is necessary to compute again since 
        c_emb = self.bottleneck.linear(latent)

        if all(x=='continuous' for x in self.concept_type):
            # If the concepts are continuous, we perform: c_emb * c_pred
            c_emb = self.continuous_mix(c_emb, input_concepts)
        else:
            c_emb = concept_embedding_mixture(c_emb, input_concepts)

        # adding memory dimension to concept weights
        c_weights = self.concept_relevance(c_emb).unsqueeze(dim=1)
        self.__predicted_weights = c_weights

        y_bias = None
        if self.use_bias:
            # adding memory dimension to bias
            y_bias = self.bias_predictor(c_emb).unsqueeze(dim=1)
            self.__predicted_bias = y_bias

        y_hat = CF.linear_equation_eval(c_weights, input_concepts, y_bias)
        return {
            'y_hat': y_hat[:, :, 0],
            'c_hat': c_hat,
            'weights': c_weights,
            'y_bias': y_bias,
        }

    def loss(self, y_hat, y, c_hat=None, c=None, *args, **kwargs):
        loss = self.concept_based_loss(y_hat, y, c_hat, c)
        # adding l1 regularization to the weights
        w_loss = self.weight_reg * self.__predicted_weights.norm(p=2)
        loss += w_loss
        # adding l2 regularization to the biases if used
        if self.use_bias:
            b_loss = self.bias_reg * self.__predicted_bias.norm(p=1)
            loss += b_loss
        return loss

    def get_symbolic_equivalent(self, log_dir=None):
        """
        Returns the equation associated to the predictor of the model
        """
        
        # Return the most complex linear equation that can obtained after training (all concepts are relevant) 
        equation = linear_classifier_expression(len(self.c_names), include_bias=self.use_bias)
        store_eq(equation, log_dir)
