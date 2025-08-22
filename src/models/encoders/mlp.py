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

    def __init__(self, input_size, output_size, input_transform=None,
                 hidden_size=64, activation='ReLU', **kwargs):
        super().__init__(input_size, output_size, input_transform)
        
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            getattr(nn, activation)(),
            nn.Linear(hidden_size, output_size),
            getattr(nn, activation)(),
        )

    def forward(self, x):
        if self.input_transform is not None:
            x = self.input_transform(x)
        return self.mlp(x)