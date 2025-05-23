import torch
import torch.nn as nn
from src.models.base import BaseModel
from src.models.modules import LinearMemoryClassifier
import torch_concepts.nn as pyc_nn

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
                 latent_size = 64,
                 c_groups=None,
                 memory_size=7,
                 weight_reg=1e-4,
                 mc_approx=10,
                 ):

        super().__init__(
                 input_size, 
                 output_size,
                 task,
                 activation,
                 latent_size,
                 c_groups
                 )

        # Parameters in common with the other Concept Embedding
        # based models.
        self.embedding_size = embedding_size
        self.task_penalty = task_penalty
        self.c_names = list(c_names)
        self.int_prob = int_prob
        self.int_idxs = int_idxs
        self.has_concepts = True
        self.noise = noise
        self.concept_loss_form = nn.BCELoss()

        self.bottleneck = pyc_nn.ConceptEmbeddingBottleneck(
            latent_size,
            self.c_names,
            embedding_size,
        )
        
        self.classifier = LinearMemoryClassifier(
            output_size,
            c_names,
            y_names,
            task_penalty,
            activation=activation,
            embedding_size=embedding_size,
            latent_size=latent_size,
            memory_size=memory_size,
            weight_reg=weight_reg,
            mc_approx=mc_approx
        )
    
    def forward(self, input):  
        x, c_true, int_idxs = self.encode(input)
        
        c_emb, c_dict = self.bottleneck(
            x,
            c_true=c_true,
            intervention_idxs=int_idxs,
            intervention_rate=1.,
        )
        c_pred = c_dict['c_int']

        # Predict a CBM from the memory bank and classify the samples in the batch
        y_pred, c_pred, predicted_cbm = self.classifier(c_emb, c_pred, self.current_epoch)
        return y_pred, c_pred, predicted_cbm

    def loss(self, y_hat, y, c_hat=None, c=None):
        loss = self.concept_based_loss(y_hat, y, c_hat, c)
        loss += self.classifier.sparsity_loss()
        return loss
    
    def filter_output_for_metric(self, y_output, c_output=None):
        # Average over the last dimension, which contains the samples
        # form the Monte Carlo approximation.
        y_output = y_output.mean(dim=-1)
        return y_output, c_output
    
    def filter_output_for_loss(self, y_output, c_output=None, predicted_cbm=None):
        # This models return the predicted CBM in addition to the usual
        # y and c predictions. The loss function needs only y and c to be computed.
        return y_output, c_output
    
