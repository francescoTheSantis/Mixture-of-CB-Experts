import torch
import torch.nn as nn
import sympy
from sympy import symbols, sympify, preorder_traversal
import re
from typing import Dict, List, Tuple, Any
import numpy as np


class TrainableEquation(nn.Module):
    """
    A PyTorch module that wraps a sympy equation and makes its affine parameters trainable.
    Affine parameters are constants that appear in additive or multiplicative positions
    (but NOT in exponents).
    """
    
    def __init__(self, sympy_expr, variable_names: List[str], eps=1e-8):
        """
        Args:
            sympy_expr: A sympy expression
            variable_names: List of variable names (e.g., ['c0', 'c1', 'c2'])
            eps: Small constant to avoid numerical issues (e.g., log(0), sqrt(negative))
        """
        super(TrainableEquation, self).__init__()
        self.sympy_expr = sympy_expr
        self.variable_names = variable_names
        self.eps = eps
        
        # Extract parameters and create the parameterized expression
        self.param_map, self.parameterized_expr = self._extract_and_parameterize(sympy_expr)
        
        # Create trainable parameters
        self.params = nn.ParameterDict()
        for param_name, param_value in self.param_map.items():
            self.params[param_name] = nn.Parameter(torch.tensor(float(param_value), dtype=torch.float32))
        
        # Store the original expression for reference
        self.original_expr = str(sympy_expr)
        
    def _extract_and_parameterize(self, expr) -> Tuple[Dict[str, float], Any]:
        """
        Extract all affine parameters from the expression and replace them with symbolic parameters.
        Parameters in exponents are NOT made trainable.
        
        Returns:
            param_map: Dictionary mapping parameter names (p0, p1, ...) to their values
            parameterized_expr: Expression with numbers replaced by parameter symbols
        """
        param_map = {}
        param_counter = [0]  # Use list to allow modification in nested function
        
        def replace_numbers(node, in_exponent=False):
            """Recursively replace numbers with parameter symbols."""
            # Check if this node is a Power operation
            if node.func == sympy.Pow:
                base = node.args[0]
                exponent = node.args[1]
                # Process base normally, but mark exponent
                new_base = replace_numbers(base, in_exponent=False)
                new_exponent = replace_numbers(exponent, in_exponent=True)
                return sympy.Pow(new_base, new_exponent)
            
            # If it's a number and not in an exponent, make it trainable
            elif node.is_Number and not in_exponent:
                param_name = f'p{param_counter[0]}'
                param_map[param_name] = float(node)
                param_counter[0] += 1
                return sympy.Symbol(param_name)
            
            # If it's a symbol (variable or already a parameter), keep it
            elif node.is_Symbol:
                return node
            
            # If it's a function or operation, process its arguments
            elif node.is_Function or len(node.args) > 0:
                new_args = [replace_numbers(arg, in_exponent=in_exponent) for arg in node.args]
                return node.func(*new_args) if new_args else node
            
            # Otherwise, return as is
            else:
                return node
        
        parameterized_expr = replace_numbers(expr)
        return param_map, parameterized_expr
    
    def _safe_operation(self, operation, x, min_val=None, max_val=None):
        """
        Apply operation with safety checks to avoid NaN/Inf.
        
        Args:
            operation: The operation to apply (e.g., torch.log, torch.sqrt)
            x: Input tensor
            min_val: Minimum value to clamp x to
            max_val: Maximum value to clamp x to
        """
        if min_val is not None:
            x = torch.clamp(x, min=min_val)
        if max_val is not None:
            x = torch.clamp(x, max=max_val)
        
        result = operation(x)
        # Replace any NaN or Inf with 0
        result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
        return result
    
    def _sympy_to_torch(self, expr, var_dict):
        """
        Convert a sympy expression to a torch computation.
        
        Args:
            expr: Sympy expression (with parameters as symbols)
            var_dict: Dictionary mapping variable/parameter names to torch tensors
        """
        if expr.is_Symbol:
            return var_dict[str(expr)]
        
        elif expr.is_Number:
            return torch.tensor(float(expr), dtype=torch.float32, device=next(iter(var_dict.values())).device)
        
        # Handle operations
        elif expr.is_Add:
            result = self._sympy_to_torch(expr.args[0], var_dict)
            for arg in expr.args[1:]:
                result = result + self._sympy_to_torch(arg, var_dict)
            return result
        
        elif expr.is_Mul:
            result = self._sympy_to_torch(expr.args[0], var_dict)
            for arg in expr.args[1:]:
                result = result * self._sympy_to_torch(arg, var_dict)
            return result
        
        elif expr.func == sympy.Pow:
            base = self._sympy_to_torch(expr.args[0], var_dict)
            exponent = self._sympy_to_torch(expr.args[1], var_dict)
            
            # Handle division (negative exponents): x^(-n) = 1/x^n
            # Add epsilon to avoid division by zero
            if expr.args[1].is_Number and float(expr.args[1]) < 0:
                base = torch.clamp(torch.abs(base), min=self.eps)
            # Handle negative bases with non-integer exponents
            elif not expr.args[1].is_Integer:
                base = torch.abs(base) + self.eps
            
            return torch.pow(base, exponent)
        
        # Trigonometric functions
        elif expr.func == sympy.sin:
            return torch.sin(self._sympy_to_torch(expr.args[0], var_dict))
        
        elif expr.func == sympy.cos:
            return torch.cos(self._sympy_to_torch(expr.args[0], var_dict))
        
        elif expr.func == sympy.tan:
            arg = self._sympy_to_torch(expr.args[0], var_dict)
            # Clamp to avoid tan going to infinity
            arg = torch.clamp(arg, min=-np.pi/2 + 0.01, max=np.pi/2 - 0.01)
            return torch.tan(arg)
        
        # Exponential and logarithm
        elif expr.func == sympy.exp:
            arg = self._sympy_to_torch(expr.args[0], var_dict)
            # Clamp to avoid overflow
            arg = torch.clamp(arg, max=20)
            return torch.exp(arg)
        
        elif expr.func == sympy.log:
            arg = self._sympy_to_torch(expr.args[0], var_dict)
            return self._safe_operation(torch.log, arg, min_val=self.eps)
        
        # Square root
        elif expr.func == sympy.sqrt:
            arg = self._sympy_to_torch(expr.args[0], var_dict)
            return self._safe_operation(torch.sqrt, arg, min_val=0)
        
        # Absolute value
        elif expr.func == sympy.Abs:
            return torch.abs(self._sympy_to_torch(expr.args[0], var_dict))
        
        # Inverse trigonometric
        elif expr.func == sympy.asin:
            arg = self._sympy_to_torch(expr.args[0], var_dict)
            arg = torch.clamp(arg, min=-1 + self.eps, max=1 - self.eps)
            return torch.asin(arg)
        
        elif expr.func == sympy.acos:
            arg = self._sympy_to_torch(expr.args[0], var_dict)
            arg = torch.clamp(arg, min=-1 + self.eps, max=1 - self.eps)
            return torch.acos(arg)
        
        elif expr.func == sympy.atan:
            return torch.atan(self._sympy_to_torch(expr.args[0], var_dict))
        
        elif expr.func == sympy.tanh:
            return torch.tanh(self._sympy_to_torch(expr.args[0], var_dict))

        else:
            raise NotImplementedError(f"Function {expr.func} not implemented")
    
    def forward(self, **variables):
        """
        Evaluate the equation with given variable values.
        
        Args:
            **variables: Keyword arguments for each variable (e.g., c0=tensor, c1=tensor)
        
        Returns:
            Tensor with the evaluated equation
        """
        # Create a combined dictionary with both variables and parameters
        var_dict = {}
        
        # Add input variables
        for var_name in self.variable_names:
            if var_name in variables:
                var_dict[var_name] = variables[var_name]
            else:
                raise ValueError(f"Variable {var_name} not provided")
        
        # Add trainable parameters
        for param_name, param_value in self.params.items():
            var_dict[param_name] = param_value
        
        # Evaluate the expression
        result = self._sympy_to_torch(self.parameterized_expr, var_dict)
        
        return result
    
    def get_param_values(self):
        """Return a dictionary of current parameter values."""
        return {name: param.item() for name, param in self.params.items()}
    
    def get_equation_string(self):
        """Get the current equation with parameter values substituted."""
        param_values = self.get_param_values()
        expr = self.parameterized_expr
        for param_name, param_value in param_values.items():
            expr = expr.subs(sympy.Symbol(param_name), param_value)
        return str(expr)


class SymbolicPredictor(nn.Module):
    """
    A predictor that uses a dictionary of trainable equations organized by sets.
    Similar to PriorPredictor but with trainable parameters.
    """
    
    def __init__(self, 
                 equations: Dict[str, Dict[str, Any]],
                 c_names: List[str],
                 eps: float = 1e-8):
        """
        Args:
            equations: Dictionary of equation sets.
                      Structure: {
                          'set1': {'name1': sympy_equation1, 'name2': sympy_equation2},
                          'set2': {'name1': sympy_equation3, ...},
                          ...
                      }
                      Values can be either sympy expressions or strings that will be parsed.
            c_names: List of concept names
            eps: Small constant for numerical stability
        """
        super(SymbolicPredictor, self).__init__()
        self.equations = equations
        self.c_names = c_names
        self.eps = eps
        self.show_explanations = False
        self.equations_for_explanations_ready = False
        
        if equations is None or len(equations) == 0:
            raise ValueError("Equations dictionary cannot be None or empty.")
        
        # Validate structure: all sets should have the same number of equations
        set_sizes = [len(eq_dict) for eq_dict in equations.values()]
        if len(set(set_sizes)) != 1:
            raise ValueError("All equation sets must have the same number of equations.")
        
        self.memory_size = len(equations)  # Number of sets
        self.task_size = set_sizes[0]  # Number of equations per set
        
        # Define variable names
        if self.c_names is None:
            self.variable_names = [f'c{i}' for i in range(len(c_names))]
        else:
            self.variable_names = self.c_names

        # Prepare equations: convert strings to trainable modules
        self._prepare_equations()
    
    def _prepare_equations(self):
        """
        Convert sympy equations to TrainableEquation modules.
        Structure: {set_name: {eq_name: TrainableEquation}}
        """
        self.trainable_equations = nn.ModuleDict()
        self.equation_names = {}
        
        for set_name, eq_dict in self.equations.items():
            str_set_name = str(set_name)
            self.trainable_equations[str_set_name] = nn.ModuleDict()
            self.equation_names[str_set_name] = []
            
            for eq_name, eq_input in eq_dict.items():
                # Convert to sympy expression if it's a string
                if isinstance(eq_input, str):
                    sympy_expr = sympify(eq_input)
                else:
                    # Assume it's already a sympy expression
                    sympy_expr = eq_input
                
                # Create trainable equation
                trainable_eq = TrainableEquation(
                    sympy_expr=sympy_expr,
                    variable_names=self.variable_names,
                    eps=self.eps
                )
                
                self.trainable_equations[str_set_name][eq_name] = trainable_eq
                self.equation_names[str_set_name].append(eq_name)
    
    def forward(self, prob_per_classifier, input_concepts, *args, **kwargs):
        """
        Execute equations based on the selected set and aggregate results.
        
        Args:
            prob_per_classifier: Probability distribution over sets/memory (batch, memory_size, task_size)
            input_concepts: Input concept values (batch, n_concepts)
        
        Returns:
            Dictionary with:
                - y_hat: Predicted outputs (batch, task_size, n_samples)
                - explanations: Equation explanations (if enabled)
        """
        batch_size = input_concepts.shape[0]
        
        # Create variable dictionary for equation evaluation
        var_dict = {
            self.variable_names[i]: input_concepts[:, i] 
            for i in range(input_concepts.shape[1])
        }
        
        # Execute equations for each set
        memory_outputs = []
        
        for set_idx, set_name in enumerate(sorted(self.trainable_equations.keys())):
            task_outputs = []
            
            # Execute each equation in this set
            for eq_name in self.equation_names[set_name]:
                eq_module = self.trainable_equations[set_name][eq_name]
                
                # Evaluate equation
                eq_output = eq_module(**var_dict)
                
                # Ensure output has correct shape (batch_size,)
                if eq_output.dim() == 0:  # Scalar constant
                    eq_output = eq_output.unsqueeze(0).expand(batch_size)
                elif eq_output.dim() > 1:
                    eq_output = eq_output.squeeze()
                
                if eq_output.dim() == 0:
                    eq_output = eq_output.unsqueeze(0).expand(batch_size)
                
                task_outputs.append(eq_output)
            
            # Stack outputs for this set along task dimension
            # Shape: (batch, task_size)
            set_output = torch.stack(task_outputs, dim=1)
            memory_outputs.append(set_output)
        
        # Stack all set outputs
        # Shape: (batch, memory_size, task_size)
        eq_outputs = torch.stack(memory_outputs, dim=1)
        
        # Aggregate using probability distribution
        # y_hat shape: (batch, task_size, n_samples)
        y_hat = torch.einsum('bms,bmt->bts', prob_per_classifier, eq_outputs)
        
        # Get explanations
        if self.training:
            explanations = None
        else:
            explanations = self._get_explanations(prob_per_classifier, y_hat)
        
        return {
            'y_hat': y_hat,
            'explanations': explanations
        }
    
    def _get_explanations(self, prob_per_classifier, y_hat):
        """Get equation explanations based on selected equations."""
        if self.show_explanations:
            if not self.equations_for_explanations_ready:
                self._setup_string_equations()
            # TODO: Implement detailed explanations
            explanations = None
        else:
            explanations = [None] * prob_per_classifier.size(0)
        
        return explanations
    
    def _setup_string_equations(self):
        """Setup string representations of equations for explanations."""
        self.equations_for_explanations_ready = True
        self.string_equations = {}
        
        for set_name in self.trainable_equations.keys():
            self.string_equations[set_name] = {}
            for eq_name, eq_module in self.trainable_equations[set_name].items():
                self.string_equations[set_name][eq_name] = eq_module.get_equation_string()

    def get_learned_equations(self, log_dir=None, fine_tuned=False):
        """
        Get the current equations with learned parameter values.
        
        Args:
            log_dir: Directory to save equations (optional)
        
        Returns:
            Dictionary of learned equations
        """
        learned_equations = {}
        
        for set_name in self.trainable_equations.keys():
            learned_equations[set_name] = {}
            for eq_name, eq_module in self.trainable_equations[set_name].items():
                learned_equations[set_name][eq_name] = eq_module.get_equation_string()
        
        # Save to file if log_dir is provided
        if log_dir is not None:
            import os
            os.makedirs(log_dir, exist_ok=True)
            name = "tuned_learned_equations" if fine_tuned else "learned_equations"
            with open(f"{log_dir}/{name}.txt", "w") as f:
                for set_name, eq_dict in learned_equations.items():
                    f.write(f"Set: {set_name}\n")
                    for eq_name, eq_str in eq_dict.items():
                        f.write(f"  {eq_name}: {eq_str}\n")
                    f.write("\n")
        
        return learned_equations
    
    def get_parameter_summary(self):
        """
        Get a summary of all trainable parameters.
        
        Returns:
            Dictionary with parameter information
        """
        summary = {}
        
        for set_name in self.trainable_equations.keys():
            summary[set_name] = {}
            for eq_name, eq_module in self.trainable_equations[set_name].items():
                summary[set_name][eq_name] = {
                    'original': eq_module.original_expr,
                    'parameters': eq_module.get_param_values(),
                    'current': eq_module.get_equation_string()
                }
        
        return summary
