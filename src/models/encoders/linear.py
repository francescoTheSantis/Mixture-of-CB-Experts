from src.models.encoders.base import BaseEncoder
import torch
from torch import nn
import sympy as sp

class LinearEncoder(BaseEncoder):
    """
    A simple linear encoder that can be used as a base for concept models.

    Args:
        input_size (int): Number of input features.
        output_size (int): Number of output targets.
        activation (str): Activation function to use in the encoder.
    """

    def __init__(self, input_size, output_size, input_transform=None, activation='ReLU'):
        super().__init__(input_size, output_size, input_transform)
        if input_transform is not None:
            self.input_transform.flatten = True
        if output_size == -1:
            self.linear = torch.nn.Identity()
            self.activation = torch.nn.Identity()
        else:
            self.linear = nn.Linear(input_size, output_size)

            self.activation = getattr(nn, activation)()

    def forward(self, x):
        if self.input_transform is not None:
            x = self.input_transform(x)
        return self.activation(self.linear(x))
    
    def to_symbolic(self):
        """
        Generates a symbolic expression (SymPy object) representing
        the forward computation of the linear encoder as scalar equations.
        
        For each output neuron, creates: y_j = activation(sum_i(w_ji * x_i) + b_j)
        
        Returns:
            list or sympy.Expr: List of symbolic expressions for each output neuron,
                               or single expression if output_size == 1
            
        Example:
            For input_size=2, output_size=1 with ReLU:
            y_1 = ReLU(w_1 * x_1 + w_2 * x_2 + b_1)
        """
        # Handle identity case
        if isinstance(self.linear, torch.nn.Identity):
            # Return list of input variables
            return [sp.Symbol(f'x_{i+1}') for i in range(self.input_size)]
        
        # Get activation function
        def apply_activation(expr):
            if isinstance(self.activation, torch.nn.Identity):
                return expr
            elif isinstance(self.activation, torch.nn.ReLU):
                return sp.Function('ReLU')(expr)
            elif isinstance(self.activation, torch.nn.Sigmoid):
                return sp.Function('Sigmoid')(expr)
            elif isinstance(self.activation, torch.nn.Tanh):
                return sp.Function('tanh')(expr)
            elif isinstance(self.activation, torch.nn.LeakyReLU):
                return sp.Function('LeakyReLU')(expr)
            elif isinstance(self.activation, torch.nn.ELU):
                return sp.Function('ELU')(expr)
            elif isinstance(self.activation, torch.nn.GELU):
                return sp.Function('GELU')(expr)
            else:
                activation_name = self.activation.__class__.__name__
                return sp.Function(activation_name)(expr)
        
        # Create symbolic expressions for each output
        outputs = []
        for j in range(self.output_size):
            # For output neuron j: sum over all inputs
            linear_combination = []
            for i in range(self.input_size):
                w_ji = sp.Symbol(f'w_{j+1}{i+1}')
                x_i = sp.Symbol(f'x_{i+1}')
                linear_combination.append(sp.Mul(w_ji, x_i, evaluate=False))
            
            # Add bias
            b_j = sp.Symbol(f'b_{j+1}')
            linear_expr = sp.Add(*linear_combination, b_j, evaluate=False)
            
            # Apply activation
            output_expr = apply_activation(linear_expr)
            outputs.append(output_expr)
        
        # Return single expression if only one output, otherwise list
        return outputs[0] if len(outputs) == 1 else outputs
