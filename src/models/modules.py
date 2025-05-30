import torch 
import torch.nn as nn
import torch.nn.functional as F

class LinearMemoryClassifier(nn.Module):
    def __init__(self, 
                 output_size,
                 c_names,
                 y_names,
                 task_penalty,
                 activation='ReLU',
                 embedding_size = 16,
                 latent_size = 128,
                 memory_size=7,
                 weight_reg=1e-4,
                 mc_approx=10
                ):
        super().__init__()

        self.embedding_size = embedding_size
        self.latent_size = latent_size
        self.task_penalty = task_penalty
        self.c_names = list(c_names)
        self.y_names = list(y_names)
        self.weight_reg = weight_reg
        self.output_size = output_size

        # Parameters specific to this model
        self.memory_size = memory_size

        # The memory bank is composed by multiple CBMs
        self.memory_bank = nn.Parameter(
            torch.randn(memory_size, output_size, len(c_names))
        )

        # The global class biases
        self.biases = nn.Parameter(
            torch.randn(output_size)
        )

        # The selector is an MLP that selects 1 CBM from the memory bank
        self.selector = nn.Sequential(
            nn.Linear(embedding_size * len(self.c_names), latent_size),
            getattr(nn, activation)(),
            nn.Linear(latent_size, memory_size)
        )
    
        # Number of samples for the Monte-Carlo approximation
        self.mc_approx = mc_approx

    def compute_temperature(self, current_epoch, tau_init=1, tau_min=0.4, decay_rate=0.99):
        # The temperature is decayed from initial_temp to min_temp
        # over the course of the training
        tau = max(tau_min, tau_init * decay_rate ** current_epoch)
        return tau

    def classify(self, c_emb, c_pred, current_epoch=0):
        memory_bank = self.memory_bank
        bsz = c_emb.shape[0]

        # Prepare the parameters needed for the selector to sample from the memory bank
        current_tau = self.compute_temperature(current_epoch)
        c_emb = c_emb.view(bsz, -1)

        # For each sample, we select the corresponding CBM in the memory bank.
        # This implies computing the logits of the categorical distribution
        # that will be used to sample the CBM.
        selection = self.selector(c_emb)
        
        # Store a copy of the selection for metrics
        selection_dist = selection.clone()
        selection = selection.unsqueeze(-1)

        # Reshape the memory bank to match the selection
        # Dimension: (bsz, memory_size, output_size, len(c_names))
        memory_bank = memory_bank.unsqueeze(0).expand(bsz, -1, -1, -1)

        # At training time, we sample multiple times (Monte-Carlo approximation)
        # from a categorical distribution
        # At inference time, only one sample is taken
        if self.training:
           n_samples = self.mc_approx
        else:
           n_samples = 1

        selection = selection.expand(-1, -1, n_samples)
        
        # Sample from the categorical distribution using the Gumbel-Softmax
        # Dimension: (bsz, memory_size, n_samples)
        selection = F.gumbel_softmax(selection, 
                                    tau=current_tau, 
                                    hard=True,
                                    dim=1)

        # Select the CBM from the memory bank
        # Dimension: (bsz, output_size, len(c_names), n_samples)
        predicted_cbm = torch.einsum('bmtc,bms->btcs', memory_bank, selection)

        # Classify the sample by computing the matrix multiplication
        # between the selected CBM and the concept predictions
        # Dimension: (bsz, output_size, n_samples)
        expanded_c_pred = c_pred.unsqueeze(-1).expand(-1, -1, n_samples)
        y_probs = torch.einsum('bcs,btcs->bts', expanded_c_pred, predicted_cbm)

        # Add the global biases
        y_probs = y_probs + self.biases.unsqueeze(0).unsqueeze(-1).expand(bsz, -1, n_samples)

        return y_probs, predicted_cbm, selection_dist
    
    def forward(self, c_emb, c_pred, current_epoch=0):
        return self.classify(c_emb, c_pred, current_epoch)
    
    def sparsity_loss(self):
        # The sparsity loss is computed as the L1 norm of the memory bank
        loss = self.weight_reg * self.memory_bank.norm(p=1)
        return loss
