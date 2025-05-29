import torch
import torch.nn as nn
from src.models.v_cem import VariationalConceptEmbeddingModel
import torch.nn.functional as F
from _OLD.modules import LinearMemoryClassifier

class LinearMemoryReasoner(VariationalConceptEmbeddingModel):
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
                 kl_penalty=5e-2,
                 randint_epoch_start=5,
                 memory_size=7,
                 weight_reg=1e-4
                 ):

        super().__init__(
                 input_size, 
                 output_size,
                 c_names,
                 y_names,
                 task, 
                 task_penalty,
                 activation,
                 int_prob,
                 int_idxs,
                 noise,
                 embedding_size,
                 latent_size,
                 c_groups,
                 kl_penalty,
                 randint_epoch_start,
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
            weight_reg=weight_reg
        )
    
    def forward(self, input):  
        x, c_true, int_idxs = self.encode(input)
        c_emb, c_pred, mu, logvar = self.get_concept_latent(x, c_true, int_idxs)
        # Store the mu and logvar for the loss computation
        self.mu = mu
        self.logvar = logvar
        
        y_pred = self.classifier(c_emb, c_pred, self.current_epoch)
        return y_pred, c_pred

    def loss(self, y_pred, y, c_pred, c):
        loss = self.v_loss(y_pred, y, c_pred, c)
        loss += self.classifier.sparsity_loss()
        return loss
