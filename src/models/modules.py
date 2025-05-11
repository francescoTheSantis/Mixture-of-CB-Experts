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
                 weight_reg=1e-4
                ):
        super().__init__()

        self.embedding_size = embedding_size
        self.latent_size = latent_size
        self.task_penalty = task_penalty
        self.c_names = list(c_names)
        self.y_names = list(y_names)
        self.weight_reg = weight_reg
        self.output_size = output_size

        # Parameters specific to this model.
        self.memory_size = memory_size
        # The memory bank is composed by the concepts and the biases
        self.memory_bank = nn.Parameter(
            torch.randn(memory_size, output_size, len(c_names)+1)
        )

        self.selector = nn.Sequential(
            nn.Linear(embedding_size * len(self.c_names), latent_size),
            getattr(nn, activation)(),
            nn.Linear(latent_size, self.memory_size * output_size)
        )

    def compute_temperature(self, current_epoch, tau_init=1, tau_min=0.1, decay_rate=0.99):
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
        selection = self.selector(c_emb)
        selection = selection.view(-1, self.output_size, self.memory_size)

        # Reshape the memory bank to match the selection
        memory_bank = memory_bank.unsqueeze(0).expand(bsz, -1, -1, -1)

        # Sample and reshape
        selection = F.softmax(selection, dim=-1)
        #selection = F.gumbel_softmax(selection, tau=current_tau, hard=True)
        selection = selection.unsqueeze(-1).repeat(1, 1, 1, memory_bank.shape[-1])
        selection = selection.permute(0, 1, 3, 2)

        y_probs = torch.zeros(bsz, self.output_size, device=c_emb.device)
        for i in range(self.output_size):
            # Use the sampled selection to get the concepts from memory bank.
            mat_prod = torch.bmm(selection[:, i, :, :], memory_bank[:, :, i, :])
            weights_bias = torch.diagonal(mat_prod, dim1 = -2, dim2 = -1)
            # Get the bias from the memory bank
            bias = weights_bias[:, -1]
            # Get the weights from the memory bank
            weights = weights_bias[:, :-1]
            # Compute the final prediction as a weighted sum of weights and concepts
            y_probs[:,i] = torch.bmm(weights.unsqueeze(1), c_pred.unsqueeze(-1)).squeeze()
            # Add bias
            y_probs[:,i] += bias
        return y_probs
    
    def forward(self, c_emb, c_pred, current_epoch=0):
        return self.classify(c_emb, c_pred, current_epoch)
    
    def sparsity_loss(self):
        loss = self.weight_reg * self.memory_bank[:,:,:-1].norm(p=1)
        return loss