import torch
import torch.nn as nn
from src.models.base import BaseModel

class VariationalConceptEmbeddingModel(BaseModel):
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

        # Parameters specific to this model.
        self.kl_penalty = kl_penalty
        self.randint_epoch_start = randint_epoch_start

        # Initialize learnable concept prototypes using normal distribution
        self.prototype_emb_pos = nn.Parameter(
            torch.randn(len(c_names), embedding_size)
        )
        self.prototype_emb_neg = nn.Parameter(
            torch.randn(len(c_names), embedding_size)
        )

        self.concept_scorers = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.mu_layer = nn.ModuleList()
        self.logvar_layer = nn.ModuleList()

        for _ in range(len(self.c_names)):
            layers = nn.Sequential(
                nn.Linear(latent_size, 1),
                nn.Sigmoid()
            )
            self.concept_scorers.append(layers)

            layers = nn.Sequential(
                nn.Linear(latent_size + 1, latent_size + 1),
                nn.ReLU()
            )

            self.layers.append(layers)

            self.mu_layer.append(
                nn.Linear(latent_size + 1, embedding_size)
            )  

            self.logvar_layer.append(
                nn.Linear(latent_size + 1, embedding_size)
            )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_size*len(c_names), len(c_names)),
            nn.ReLU(),
            nn.Linear(len(c_names), len(c_names))
        )

    def apply_intervention_embedding(self, c_pred, c_int, c_emb, concept_idx):
        c_int = c_int.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.prototype_emb_pos.shape[-1]).int()
        cloned_c_pred = c_pred.detach().clone().unsqueeze(-1).expand(-1, -1, self.prototype_emb_pos.shape[-1])
        cloned_c_pred = torch.where(cloned_c_pred>0.5,1,0)
        prototype_emb = cloned_c_pred * self.prototype_emb_pos[None, concept_idx, :] + (1 - cloned_c_pred) * self.prototype_emb_neg[None, concept_idx, :]
        prototype_emb = prototype_emb.squeeze()
        c_int = c_int.squeeze()
        int_emb = c_int * prototype_emb + (1 - c_int) * c_emb
        return int_emb
    
    def apply_intervention_concept(self, labels, mask, predictions):
        mask = mask.int().unsqueeze(-1)
        labels = labels.unsqueeze(-1)
        int_c = labels * mask + predictions * (1-mask)
        return int_c

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def classify(self, c_emb):
        c_emb = c_emb.view(c_emb.shape[0], -1)
        y_pred = self.classifier(c_emb)
        return y_pred

    def get_concept_latent(self, x, c, int_idxs=None):
        c_pred_list, c_emb_list, mu_list, logvar_list = [], [], [], []
        # x = self.shared_layers(x)
        for i in range(len(self.c_names)):
            c_pred = self.concept_scorers[i](x) 
            emb = self.layers[i](torch.cat([x, c_pred], dim=-1))
            mu = self.mu_layer[i](emb)
            logvar = self.logvar_layer[i](emb)
            if self.training:
                c_emb = self.reparameterize(mu, logvar)
            else:
                c_emb = mu
            #if self.int_prob!=None and self.current_epoch>self.randint_epoch_start:
            if self.current_epoch>self.randint_epoch_start or not self.training:
                c_pred = self.apply_intervention_concept(c[:,i], int_idxs[:,i], c_pred)
                c_emb = self.apply_intervention_embedding(c_pred, int_idxs[:,i], c_emb, i)
            c_emb_list.append(c_emb.unsqueeze(1))
            c_pred_list.append(c_pred.unsqueeze(1))
            mu_list.append(mu.unsqueeze(1))
            logvar_list.append(logvar.unsqueeze(1))
        c_emb = torch.cat(c_emb_list, dim=1) 
        c_pred = torch.cat(c_pred_list, dim=1)[:,:,0] 
        mu = torch.cat(mu_list, dim=1) 
        logvar = torch.cat(logvar_list, dim=1)
        return c_emb, c_pred, mu, logvar
    
    def forward(self, input):  
        x, c_true, int_idxs = self.encode(input)
        c_emb, c_pred, mu, logvar = self.get_concept_latent(x, c_true, int_idxs)
        y_pred = self.classify(c_emb)
        # Store the mu and logvar for the loss computation
        self.mu = mu
        self.logvar = logvar
        return y_pred, c_pred

    def D_kl_gaussian(self, mu_q, logvar_q, mu_p):
        dot_prod = torch.bmm((mu_q - mu_p), (mu_q - mu_p).permute(0,2,1)).diagonal(dim1=-2, dim2=-1)
        d_kl = 0.5 * (dot_prod - self.embedding_size - logvar_q.sum(dim=-1) + logvar_q.exp().sum(dim=-1)) 
        return d_kl.mean() # average over the batch and concepts dimension

    def filter_output_for_loss(self, y_output, c_output=None):
        return y_output, c_output

    def v_loss(self, y_pred, y, c_pred, c):
        # Concept + Task loss
        concept_task_loss = self.concept_based_loss(y_pred, y, c_pred, c)

        # KL divergence
        cloned_c_pred = c_pred.detach().clone().unsqueeze(-1).expand(-1, -1, self.embedding_size)
        prototype_emb = cloned_c_pred * self.prototype_emb_pos[None, :, :] + (1 - cloned_c_pred) * self.prototype_emb_neg[None, :, :]
        D_kl = self.D_kl_gaussian(self.mu, self.logvar, prototype_emb)

        loss = concept_task_loss + self.kl_penalty * D_kl
        return loss
    
    def loss(self, y_pred, y, c_pred, c):
        return self.v_loss(y_pred, y, c_pred, c)
