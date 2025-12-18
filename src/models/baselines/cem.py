import torch
import torch.nn as nn
import torch_concepts.nn as pyc_nn
from torch_concepts.nn import concept_embedding_mixture
from src.models.encoders.mlp import MLPEncoder
from src.utils.expression_utils import store_eq
from src.models.baselines.base import BaseModel

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

    def continuous_mix(self, c_emb, c_pred):
        """
        Create the concept embedding when the concepts are continuous
        """
        # We take only the first embedding produced by the concept embedding layer:
        c_emb = c_emb[:,:,:self.embedding_size] # (batch_size, num_concepts)
        c_emb = c_emb * c_pred[:,:,None]  # (batch_size, num_concepts, embedding_size)
        return c_emb

    def forward(self, input):
        x, _, c_true, int_idxs = self.encode(input)

        _, c_dict = self.bottleneck(
            x,
            c_true=c_true,
            intervention_idxs=int_idxs,
            intervention_rate=1.,
        )
        c_hat = c_dict['c_int']

        c_hat, input_concepts = self._process_concepts(c_hat, c_true, int_idxs)

        # It is necessary to compute again since the embeddings 
        # may have changed due to the interventions
        c_emb = self.bottleneck.linear(x)

        if all(x=='continuous' for x in self.concept_type):
            # If the concepts are continuous, we perform: c_emb * c_pred
            c_emb = self.continuous_mix(c_emb, input_concepts)
        else:
            c_emb = concept_embedding_mixture(c_emb, input_concepts)

        y_hat = self.y_predictor(c_emb.flatten(-2))
        return {
            'y_hat': y_hat,
            'c_hat': c_hat
        }

    def loss(self, y_hat, y, c_hat=None, c=None, *args, **kwargs):
        loss = self.concept_based_loss(y_hat, y, c_hat, c)
        return loss

    def get_symbolic_equivalent(self, log_dir=None, return_equations=False):
        """
        Returns the equation associated to the predictor of the model
        """

        # Get as many equations as the output size
        equations = self.y_predictor.to_symbolic()

        if return_equations:
            return equations

        # If the output is greater than 1, equations will be a list.
        # Each equation in the list will have the same complexity, therefore we return only the first one.
        if self.output_size > 1:
            store_eq(equations[0], log_dir)
        store_eq(equations, log_dir)


