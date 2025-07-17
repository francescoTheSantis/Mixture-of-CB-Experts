from torch import nn

class BaseEncoder(nn.Module):
    """
    Base class for encoders. It provides a common interface for all encoders.
    All encoders should inherit from this class and implement the `forward` method.

    Args:
        input_size (int): dimension of the input.
        output_size (int): dimension of the output.
        input_transform (callable, optional): A function to transform the input data before encoding.
    """

    def __init__(self, input_size, output_size, input_transform=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.input_transform = input_transform

    def forward(self, x):
        """
        Forward pass of the encoder.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Encoded representation of the input.
        """
        raise NotImplementedError("Forward method should be implemented by subclasses.")