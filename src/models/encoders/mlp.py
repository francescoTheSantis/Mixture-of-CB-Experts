from src.models.encoders.base import BaseEncoder
from torch import nn

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

