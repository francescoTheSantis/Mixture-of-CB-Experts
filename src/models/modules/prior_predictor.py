import torch
import torch.nn as nn
import sympy
import sympytorch
import omegaconf

from src.utils.expression_utils import store_eq

class PriorPredictor(nn.Module):
    def __init__(self, 
                 equations: dict[int, list[str]],  # Changed to dict[memory_index, list of equations]
                 c_names: list[str]
                 ):
        super(PriorPredictor, self).__init__()
        self.equations = equations
        self.c_names = c_names
        self.show_explanations = False
        self.equations_for_explanations_ready = False

        if equations is None or len(equations) == 0:
            raise ValueError("Equations dictionary cannot be None or empty.")

        task_mem = []
        for k, v in equations.items():
            assert isinstance(k, str), "Equation keys must be strings representing memory indices."
            assert isinstance(v, (list, omegaconf.listconfig.ListConfig)), "Equation values must be lists or ListConfig."

            task_mem.append(len(v))  # Number of memory indices

        # check if task_mem has consistent number of equations per memory index
        if len(set(task_mem)) != 1:
            raise ValueError("Inconsistent number of equations per memory index.")
        self.memory_size = task_mem[0]

        self._prepare_equations(equations)

    def _prepare_equations(self, equations):
        # Dictionary to store equations organized by memory index
        # Structure: {memory_index: [(torch_eq, sympy_vars, sympy_eq), ...]}
        self.torch_equations = {}
        self.sympy_variables = {}
        self.sympy_equations = {}

        # define the variables
        variables = [f'c{i}' for i, name in enumerate(self.c_names)]
        self.string_variables = variables

        # Convert the string equations to torch functions for each memory index
        for memory_idx, eq_list in equations.items():
            self.torch_equations[memory_idx] = []
            self.sympy_variables[memory_idx] = []
            self.sympy_equations[memory_idx] = []
            
            for eq in eq_list:
                self._convert_equation_to_torch(eq, variables, memory_idx)

    def _convert_equation_to_torch(self, equation_str, variables, memory_idx):
        # Define the symbols (variables)
        sympy_vars = sympy.symbols(variables)
        # Define the equation in a textual format
        str_exp = equation_str
        # Convert to sympy expression
        sympy_exp = sympy.sympify(str_exp)
        # Convert the textual equation into an executable PyTorch module
        torch_exp = sympytorch.SymPyModule(expressions=[sympy_exp])
        # Store the torch module, the sympy variables, and the sympy expression 
        self.torch_equations[memory_idx].append(torch_exp)
        self.sympy_variables[memory_idx].append(sympy_vars)
        self.sympy_equations[memory_idx].append(sympy_exp)
        
    def _setup_string_equations(self):
        # If set to true, this function will never be called again
        self.equations_for_explanations_ready = True
        equations = self.known_equations
        self.string_equations = [str(eq) for eq in equations]
    
    def _get_explanations(self, prob_per_classifier, y_hat):
        # TODO: to be completed once the kan implementation is stable
        if self.show_explanations:
            if not self.equations_for_explanations_ready:
                self._setup_string_equations()
            explanations = None
        else:
            explanations = [None] * prob_per_classifier.size(0)

        return explanations
    
    def forward(self, prob_per_classifier, input_concepts, *args, **kwargs):
        # Execute all the equations for each memory index
        # Output shape: [batch, memory_size, task_size]
        # Assumes all memory indices have the same task_size (number of equations)
        batch_size = input_concepts.shape[0]
        
        # Initialize list to store outputs for each memory index
        memory_outputs = []
        
        # Process each memory index
        for memory_idx in range(self.memory_size):
            eq_list = self.torch_equations[str(memory_idx)]
            task_outputs = []
            
            # Execute each equation in the task list
            for equation in eq_list:
                # Create a dictionary mapping variable names to their values
                var_dict = dict(zip(self.string_variables, 
                                  [input_concepts[:, i] for i in range(input_concepts.shape[1])]))
                # Execute the equation function with the mapped variables
                eq_output = equation(**var_dict)
                
                # Ensure output has batch dimension (handle scalar constants)
                if eq_output.dim() == 0:  # Scalar constant
                    eq_output = eq_output.unsqueeze(0).expand(batch_size)
                elif eq_output.dim() == 1 and eq_output.shape[0] != batch_size:
                    # If it's a single value, expand it to batch size
                    eq_output = eq_output.expand(batch_size)
            
                # Ensure the output is 1D with shape (batch_size,)
                eq_output = eq_output.squeeze()
                if eq_output.dim() == 0:
                    eq_output = eq_output.unsqueeze(0).expand(batch_size)
                
                task_outputs.append(eq_output)
            
            # Stack outputs for this memory index along task dimension
            # Shape: (batch, task_size)
            memory_task_output = torch.stack(task_outputs, dim=1)
            memory_outputs.append(memory_task_output)
        
        # Stack all memory outputs
        # Shape: (batch, memory_size, task_size)
        eq_outputs = torch.stack(memory_outputs, dim=1)

        y_hat = torch.einsum('bms,bmt->bts', prob_per_classifier, eq_outputs)

        # Get the explanations (the selected equations)
        if self.training:
            explanations = None
        else:
            explanations = self._get_explanations(prob_per_classifier, y_hat)
        return {
            'y_hat':y_hat, 
            'explanations': explanations
        }
    
    def get_symbolic_equivalent(self, log_dir):
        """
        Save the sympy equations 
        """
        # For this specific class, we have multiple equations (one per memory index)
        # 
        for i in range(self.memory_size):
            store_eq(self.sympy_equations[i], log_dir=log_dir, idx=i)
