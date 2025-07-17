from src.models.encoders.base import BaseEncoder
import torch
from torch import nn

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
