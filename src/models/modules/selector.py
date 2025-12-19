import torch.nn as nn
from src.models.encoders.mlp import MLPEncoder
from src.models.encoders.linear import LinearEncoder
from torch.nn import functional as F
import numpy as np

class SelectorModel(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 n_outputs,
                 model_type='linear', 
                 activation='ReLU',
                 decay_rate='cosine'
                 ):
        super(SelectorModel, self).__init__()
        self.model_type = model_type
        self.decay_rate = decay_rate
        self.n_outputs = n_outputs
        self.memory_size = output_size
        # check decay_rate
        if self.decay_rate not in ['linear', 'exp', 'cosine']:
            raise ValueError(f"Unknown decay rate: {self.decay_rate}")

        # Output size is memory_size * n_outputs for independent selection per output
        total_output_size = output_size * n_outputs
        
        if self.model_type == 'linear':
            self.selector = LinearEncoder(
                input_size=input_size,
                output_size=total_output_size,
                activation=activation,
            )
        elif self.model_type == 'mlp':
            self.selector = MLPEncoder(
                input_size=input_size,
                output_size=total_output_size,
                hidden_size=input_size,
                activation=activation,
            )
        else:
            raise ValueError(f"Unknown selector model: {self.selector_model}")

    def compute_tau(self, global_step, tau_init=2, tau_min=0.05, decay_rate=0.99):
        if self.decay_rate == 'linear':
            # Linear decay to decrease tau over time
            tau = max(tau_min, tau_init - decay_rate * global_step)
        elif self.decay_rate == 'exp':
            # Exponential decay to decrease tau over time
            tau = max(tau_min, tau_init * decay_rate ** global_step)
        elif self.decay_rate == 'cosine':
            # Cosine decay to decrease tau over time
            tau = tau_min + (tau_init - tau_min) * (1 + np.cos(np.pi * global_step / 10000)) / 2
        else:
            raise ValueError(f"Unknown decay rate: {self.decay_rate}")
        return tau

    def forward(self, x, mc_approx=1, global_step=0):
        bsz = x.size(0)
        
        # Dimension : (bsz, memory_size)
        selector_logits = self.selector(x)

        # At training time, we sample multiple times (Monte-Carlo approximation)
        # from a categorical distribution.
        # At inference time, only one sample is taken
        if self.training:
            n_samples = mc_approx
        else:
            n_samples = 1

        # Reshape to : (bsz, memory_size, n_outputs)
        selector_logits = selector_logits.view(bsz, self.memory_size, self.n_outputs) 
        selection_dist = selector_logits.clone().detach()  

        # Dimension: (bsz, memory_size, n_outputs, n_samples)
        selector_logits = selector_logits.unsqueeze(-1).expand(-1, -1, -1, n_samples)

        # Compute the temperature for the Gumbel-Softmax distribution
        current_tau = self.compute_tau(global_step)

        # Dimension: (bsz, memory_size, n_outputs, n_samples)
        selector_probs = F.gumbel_softmax(selector_logits, 
                                          tau=current_tau, 
                                          hard=True, 
                                          dim=1)

        return {
            'selector_probs': selector_probs,
            'selection_dist': selection_dist
        }