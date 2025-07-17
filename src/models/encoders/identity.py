from src.models.encoders.base import BaseEncoder

class IdentityEncoder(BaseEncoder):
    """
    An identity encoder that simply returns the input as the output.
    This can be useful for testing or when no encoding is needed.

    Args:
        input_size (int): Number of input features.
        output_size (int): Number of output targets.
        input_transform (callable, optional): A function to transform the input data before encoding.
    """

    def __init__(self, input_size, output_size, input_transform=None):
        super().__init__(input_size, output_size, input_transform)
        assert input_size == output_size, ("Input size must match output size "
                                           "for IdentityEncoder.")

    def forward(self, x):
        if self.input_transform is not None:
            x = self.input_transform(x)
        return x