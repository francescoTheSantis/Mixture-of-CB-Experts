
import torch.nn as nn
import torch
import os
from kan import KAN
from kan.utils import SYMBOLIC_LIB 

class KANPredictor(nn.Module):
    def __init__(self, 
                 widths, 
                 grid, 
                 k, 
                 memory_size, 
                 device, 
                 speed_up_training, 
                 regularize=True):
        
        super(KANPredictor, self).__init__()
        self.widths = widths
        self.grid = grid
        self.k = k
        self.memory_size = memory_size
        self.device = device
        self.speed_up_training = speed_up_training
        self.regularize = regularize
        self.show_explanations = False # TODO: se to true once the kan implementation is stable
        self.symbolic_predictors = False # When instantiated, the kan layers will not compute symbolic formulas

        kan_params = {
                'width': self.widths, 
                'grid': self.grid,
                'k': self.k,
                'device': self.device
        }

        # Instantiate as many KAN Layers as the memory size
        self.kans = nn.ModuleList()
        for i in range(self.memory_size):
            kan_params['ckpt_path'] = os.path.join(os.getcwd(), f'kan{i}_ckpt')
            kan_layer = KAN(**kan_params)
            if self.speed_up_training:
                kan_layer = kan_layer.speed()  # Sets: symbolic_enabled=False, save_act=False, auto_save=False
            if self.regularize:
                self.lamb = 0.001  # Regularization strength 
            for param in kan_layer.get_params():
                param.requires_grad = True
            self.kans.append(kan_layer)

    def setup_kan_grid(self, grid_inputs):  
        grid_inputs = grid_inputs if grid_inputs.ndim > 1 else grid_inputs.unsqueeze(1)
        # Update the grid of all KAN layers based on the provided inputs (only if out of the default grid range)  
        for kan_layer in self.kans:
            if torch.min(grid_inputs) < -1 or torch.max(grid_inputs) > 1:
                kan_layer.update_grid_from_samples(grid_inputs)

    def allow_symbolic(self, kan):
        kan.symbolic_enabled=True
        kan.save_act=True
        kan.auto_save=True
        return kan

    def prune(self):
        for i, _ in enumerate(self.kans):
            self.kans[i].to(self.device)
            self.kans[i] = self.kans[i].prune()
            # when speed_up_training is True, we need to re-allow symbolic execution
            if self.speed_up_training:
                self.kans[i] = self.allow_symbolic(self.kans[i])

    def _sync_kan_tensors_to_device(self, kan_layer):
        """
        Ensure all KAN layer tensors are on the same device as act_fun.
        This is needed for the plot() method which calls attribute().
        """
        # Get the device from act_fun (where the main computation happens)
        target_device = kan_layer.act_fun[0].grid.device
        
        # Move edge_actscale and subnode_actscale to the target device
        if hasattr(kan_layer, 'edge_actscale') and kan_layer.edge_actscale:
            kan_layer.edge_actscale = [tensor.to(target_device) for tensor in kan_layer.edge_actscale]
        
        if hasattr(kan_layer, 'subnode_actscale') and kan_layer.subnode_actscale:
            kan_layer.subnode_actscale = [tensor.to(target_device) for tensor in kan_layer.subnode_actscale]

    def get_learned_equations(self, log_dir):

        self.symbolic_predictors = True

        equations = []
        for i, kan_layer in enumerate(self.kans):

            # Get the symbolic formula
            kan_layer.auto_symbolic(lib=SYMBOLIC_LIB, r2_threshold=0)

            for param in kan_layer.parameters():
                param.requires_grad = True

            # Enable affine parameters
            for l in range(kan_layer.depth):
                exec(f'kan_layer.node_bias{[l]}.requires_grad = True')
                exec(f'kan_layer.node_scale{[l]}.requires_grad = True')
                exec(f'kan_layer.subnode_bias{[l]}.requires_grad = True')
                exec(f'kan_layer.subnode_scale{[l]}.requires_grad = True')

            # Store the equation in the corresponding list
            equations.append(kan_layer.symbolic_formula()[0][0])

            # Plot the kan layer using the authors' plotting function
            self._sync_kan_tensors_to_device(kan_layer)
            kan_layer.plot(os.getcwd())

            # Move to device
            kan_layer.to(self.device)

        # Store the equations in a text file
        with open(f"{log_dir}/kan_equations_pre_fine_tuning.txt", "w") as f:
            for i, eq in enumerate(equations):
                f.write(f"KAN Layer {i+1}: {eq}\n")

    def _get_explanations(self, prob_per_classifier, y_hat):
        # TODO: to be completed once the kan implementation is stable
        if self.show_explanations:
            if not self.equations_for_explanations_ready:
                self._setup_string_equations()
            explanations = None
        else:
            explanations = [None] * prob_per_classifier.size(0)

        return explanations

    def _setup_string_equations(self):
        # If set to true, this function will never be called again
        self.equations_for_explanations_ready = True
        equations = []
        # convert to string
        self.string_equations = [str(eq) for eq in equations]

    def forward(self, prob_per_classifier, input_concepts):
        bsz = input_concepts.shape[0]
        # Execute all the KAN layers
        eq_outputs = []
        for i, kan_layer in enumerate(self.kans):
            if self.regularize:
                eq_output = kan_layer(input_concepts, singularity_avoiding=True, y_th=1000)
            else:
                eq_output = kan_layer(input_concepts)
            eq_outputs.append(eq_output)
        # Stack the outputs along the class dimension
        eq_outputs = torch.stack(eq_outputs, dim=1) # shape: (bsz, memory_size, n_targets)

        y_hat = torch.einsum('bms,bmt->bts', prob_per_classifier, eq_outputs)

        # Get the explanations (the selected equations)
        if self.training:
            explanations = [None] * bsz
        else:
            explanations = self._get_explanations(prob_per_classifier, y_hat)

        return {
            'y_hat': y_hat,
            'explanations': explanations,
        }

    def regularization_term(self):
        reg_term = 0.0
        if self.regularize and not self.symbolic_predictors:
            for kan_layer in self.kans:
                reg_term += kan_layer.get_reg(reg_metric="edge_forward_spline_n", lamb_l1=1, lamb_entropy=2, lamb_coef=0, lamb_coefdiff=0)
            # divide by the number of kan layers (memory_size)
            reg_term = reg_term / self.memory_size
            return reg_term * self.lamb
        else:
            return reg_term