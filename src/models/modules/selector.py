import torch.nn as nn
from src.models.encoders.mlp import MLPEncoder
from src.models.encoders.linear import LinearEncoder
from torch.nn import functional as F
import numpy as np

class SelectorModel(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 model_type='linear', 
                 activation='ReLU',
                 decay_rate='cosine',
                 gumble=False
                 ):
        super(SelectorModel, self).__init__()
        self.model_type = model_type
        self.decay_rate = decay_rate
        self.gumble = gumble

        # check decay_rate
        if self.decay_rate not in ['linear', 'exp', 'cosine']:
            raise ValueError(f"Unknown decay rate: {self.decay_rate}")

        if self.model_type == 'linear':
            self.selector = LinearEncoder(
                input_size=input_size,
                output_size=output_size,
                activation=activation,
            )
        elif self.model_type == 'mlp':
            self.selector = MLPEncoder(
                input_size=input_size,
                output_size=output_size,
                hidden_size=input_size,
                activation=activation,
            )
        else:
            raise ValueError(f"Unknown selector model: {self.selector_model}")


    def compute_tau(
        self,
        global_step,
        tau_init=20.0,
        tau_min=0.05,
        decay_rate=0.99,
        cycle_steps=25,    # length of one restart cycle
        stop_reset_step=50,  # new: global step after which no restarts happen
    ):
        # Determine if we are past the stop_reset_step
        if stop_reset_step is not None and global_step >= stop_reset_step:
            step = stop_reset_step  # freeze the cycle at stop_reset_step
        else:
            step = global_step % cycle_steps

        if self.decay_rate == 'linear':
            tau = tau_init - (tau_init - tau_min) * (step / cycle_steps)

        elif self.decay_rate == 'exp':
            tau = tau_init * (decay_rate ** step)

        elif self.decay_rate == 'cosine':
            tau = tau_min + (tau_init - tau_min) * \
                (1 + np.cos(np.pi * step / cycle_steps)) / 2

        # Ensure tau never goes below tau_min
        return max(tau, tau_min)


    def forward(self, x, global_step=0):
        bsz = x.size(0)
        
        # Dimension : (bsz, memory_size)
        selector_logits = self.selector(x)

        selection_dist = selector_logits.clone().detach()    

        # Compute the temperature for the Gumbel-Softmax distribution
        current_tau = self.compute_tau(global_step)

        if self.gumble:
            # Dimension: (bsz, memory_size)
            selector_probs = F.gumbel_softmax(selector_logits, 
                                            tau=current_tau, 
                                            hard=False, 
                                            dim=1)
        else:
            # Dimension: (bsz, memory_size)
            selector_logits = selector_logits / current_tau
            selector_probs = F.softmax(selector_logits, dim=1)

        return {
            'selector_probs': selector_probs,
            'selection_dist': selection_dist
        }