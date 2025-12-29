from src.models.encoders.base import BaseEncoder
from torch import nn
import sympy as sp
import torch

class MLPEncoder(BaseEncoder):
    """
    A simple MLP encoder that can be used as a base for concept models.

    Args:
        input_size (int): Number of input features.
        output_size (int): Number of output targets.
        hidden_size (int): Number of hidden units in the MLP.
        activation (str): Activation function to use in the MLP.
    """

    def __init__(self, 
                 input_size, 
                 output_size, 
                 input_transform=None,
                 hidden_size=64, 
                 activation='ReLU', 
                 dropout=0.1,
                 num_layers=1,
                 **kwargs):
        super().__init__(input_size, output_size, input_transform)
        
        layers = []
        in_features = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(getattr(nn, activation)())
            layers.append(nn.Dropout(dropout) if dropout is not None else nn.Identity())
            in_features = hidden_size
        layers.append(nn.Linear(in_features, output_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        if self.input_transform is not None:
            x = self.input_transform(x)
        return self.mlp(x)
    
    def to_symbolic(self, input_names=None):
        """
        Generates a symbolic expression (SymPy object) representing
        the forward computation of the MLP as scalar equations.
        
        Each hidden/output neuron is represented as:
        h_j = activation(sum_i(w_ji * input_i) + b_j)
        
        Returns:
            list or sympy.Expr: List of symbolic expressions for each output neuron,
                               or single expression if output_size == 1
            
        Example:
            For input_size=1, hidden_size=2, output_size=1, num_layers=1 with ReLU:
            h_1 = ReLU(w_11 * x_1 + b_11)
            h_2 = ReLU(w_21 * x_1 + b_21)
            y_1 = ReLU(w_1 * h_1 + w_2 * h_2 + b_1)
        """
        # Get activation function
        def apply_activation(expr, module):
            if isinstance(module, nn.Identity):
                return expr
            elif isinstance(module, nn.Dropout):
                return expr  # Skip dropout
            elif isinstance(module, nn.ReLU):
                return sp.Function('ReLU')(expr)
            elif isinstance(module, nn.Sigmoid):
                return sp.Function('Sigmoid')(expr)
            elif isinstance(module, nn.Tanh):
                return sp.Function('tanh')(expr)
            elif isinstance(module, nn.LeakyReLU):
                return sp.Function('LeakyReLU')(expr)
            elif isinstance(module, nn.ELU):
                return sp.Function('ELU')(expr)
            elif isinstance(module, nn.GELU):
                return sp.Function('GELU')(expr)
            elif isinstance(module, nn.Softmax):
                return sp.Function('Softmax')(expr)
            elif isinstance(module, nn.LogSoftmax):
                return sp.Function('LogSoftmax')(expr)
            else:
                activation_name = module.__class__.__name__
                return sp.Function(activation_name)(expr)
        
        # Track current layer outputs (start with inputs)
        current_layer_size = self.input_size
        current_symbols = [sp.Symbol(f'x_{i+1}') for i in range(self.input_size)]
        
        layer_idx = 0
        
        # Iterate through the MLP layers
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                layer_idx += 1
                next_layer_size = module.out_features
                next_symbols = []
                
                # Create symbolic expression for each neuron in this layer
                for j in range(next_layer_size):
                    # Sum over all inputs to this neuron
                    linear_combination = []
                    for i in range(current_layer_size):
                        w_ji = sp.Symbol(f'w_{layer_idx}_{j+1}{i+1}')
                        linear_combination.append(sp.Mul(w_ji, current_symbols[i], evaluate=False))
                    
                    # Add bias
                    b_j = sp.Symbol(f'b_{layer_idx}_{j+1}')
                    linear_expr = sp.Add(*linear_combination, b_j, evaluate=False)
                    
                    next_symbols.append(linear_expr)
                
                # Update for next iteration
                current_symbols = next_symbols
                current_layer_size = next_layer_size
                
            elif isinstance(module, (nn.Identity, nn.Dropout)):
                # Skip identity and dropout
                continue
                
            else:
                # Apply activation function to all current symbols
                current_symbols = [apply_activation(sym, module) for sym in current_symbols]
        
        # Return single expression if only one output, otherwise list
        return current_symbols[0] if len(current_symbols) == 1 else current_symbols

