
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
                 auto_save=False,
                 regularize=True,
                 y_names=None
                ):
        
        super(KANPredictor, self).__init__()
        self.widths = widths
        self.grid = grid
        self.k = k
        self.memory_size = memory_size
        self.device = device
        self.speed_up_training = speed_up_training
        self.regularize = regularize
        self.auto_save = auto_save
        self.show_explanations = False # TODO: se to true once the kan implementation is stable
        self.symbolic_predictors = False # When instantiated, the kan layers will not compute symbolic formulas
        self.y_names = y_names

        kan_params = {
                'width': self.widths, 
                'grid': self.grid,
                'k': self.k,
                'device': self.device,
                'auto_save': self.auto_save,
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

    def allow_symbolic(self):
        for i, kan in enumerate(self.kans):
            kan.to(self.device)
            kan.symbolic_enabled=True
            kan.save_act=True
            kan.auto_save=True
            self.kans[i] = kan
        return kan

    def prune(self):
        for i, _ in enumerate(self.kans):
            self.kans[i].to(self.device)
            self.kans[i] = self.kans[i].prune()

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

        equations = {}
        for i, kan_layer in enumerate(self.kans):
            # Get the symbolic formula
            kan_layer.auto_symbolic(lib=SYMBOLIC_LIB)

            # try:
            #     # Plot the kan layer using the authors' plotting function
            #     self._sync_kan_tensors_to_device(kan_layer)
            #     kan_layer.plot(os.getcwd(), idx=i+1) # so that the count starts from 1
            # except Exception as e:
            #     print(f"Warning: could not plot KAN layer {i} due to error: {e}")

            # Get the symbolic formulas and variable names
            equations_set = {}
            symbolic_output = kan_layer.symbolic_formula()
            learned_equations = symbolic_output[0]
            variable_names = symbolic_output[1]
            equations_set = {y_name: eq for eq, y_name in zip(learned_equations, self.y_names)}

            # Add the set of equations to the main dictionary
            equations[i] = equations_set

        # Prepare a dictionary of learned equations as strings
        str_equations = {}
        for set_name in equations.keys():
            str_equations[set_name] = {}
            for eq_name, eq_module in equations[set_name].items():
                str_equations[set_name][eq_name] = str(eq_module)

        # Save to file if log_dir is provided
        if log_dir is not None:
            import os
            os.makedirs(log_dir, exist_ok=True)
            with open(f"{log_dir}/learned_equations_kan.txt", "w") as f:
                for set_name, eq_dict in str_equations.items():
                    f.write(f"Set: {set_name}\n")
                    for eq_name, eq_str in eq_dict.items():
                        f.write(f"  {eq_name}: {eq_str}\n")
                    f.write("\n")

        return equations, variable_names

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
            eq_output = kan_layer(input_concepts, singularity_avoiding=True, y_th=1000)
            eq_outputs.append(eq_output)
        # Stack the outputs along the class dimension
        eq_outputs = torch.stack(eq_outputs, dim=1) # shape: (bsz, memory_size, n_targets)

        # With independent outputs, prob_per_classifier has shape (bsz, n_outputs, memory_size, n_samples)
        # and eq_outputs has shape (bsz, memory_size, n_outputs)
        y_hat = torch.einsum('bmys,bmy->bys', prob_per_classifier, eq_outputs)

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