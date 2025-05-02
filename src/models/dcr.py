import torch
import torch.nn as nn
import torch_concepts.nn as pyc_nn
from torch_concepts.semantic import ProductTNorm

class DeepConceptReasoner(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size,
                 c_names,
                 task, 
                 task_penalty,
                 activation='ReLU',
                 int_prob=0.1,
                 int_idxs=None,
                 noise=None,
                 embedding_size = 16,
                 latent_size = 128,
                 semantic = ProductTNorm()
                 ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.task = task
        self.embedding_size = embedding_size
        self.latent_size = latent_size
        self.task_penalty = task_penalty
        self.c_names = list(c_names)
        self.int_prob = int_prob
        self.int_idxs = int_idxs
        self.has_concepts = True
        self.noise = noise

        self.encoder = nn.Sequential(
            nn.Linear(input_size, self.latent_size),
            getattr(nn, activation)()
        )

        self.bottleneck = pyc_nn.ConceptEmbeddingBottleneck(
            latent_size,
            self.c_names,
            embedding_size,
        )
        self.y_predictor = nn.Sequential(
            nn.Linear(len(self.c_names) * embedding_size, latent_size),
            nn.LeakyReLU(),
            nn.Linear(latent_size, len(c_names)),
        )

        if task == 'classification':
            self.task_loss_form = nn.CrossEntropyLoss()
        elif task == 'regression':
            self.task_loss_form = nn.MSELoss()

        self.concept_loss_form = nn.BCELoss()

    def forward(self, input):
        x = input['x']
        c_true = input['c']
    
        # If noise is provided, create a convex combination of the input and noise
        if self.noise!=None:
            eps = torch.randn_like(x)
            x = eps * self.noise + x * (1-self.noise)
            
        x = self.encoder(x)

        # If the intervention index is not provided, 
        # all concept can be selected for interventions
        int_idxs = self.int_idxs if self.int_idxs is not None \
            else torch.ones_like(c_true).bool()
        
        c_emb, c_dict = self.bottleneck(
            x,
            c_true=c_true,
            intervention_idxs=int_idxs,
            intervention_rate=self.int_prob,
        )
        c_pred = c_dict['c_int']
        y_pred = self.y_predictor(c_emb.flatten(-2))
        return y_pred, c_pred
    
    def filter_output_for_loss(self, y_output, c_output=None):
        return y_output, c_output
    
    def loss(self, y_hat, y, c_hat=None, c=None):
        y = y.flatten().long()
        # task loss
        task_loss = self.task_loss_form(y_hat.squeeze(), y)
        # concept loss
        concept_loss = 0
        for i in range(c.shape[1]):
            concept_loss += self.concept_loss_form(c_hat[:,i], c[:,i])
        # combine the two losses
        loss = concept_loss + self.task_penalty * task_loss
        return loss


