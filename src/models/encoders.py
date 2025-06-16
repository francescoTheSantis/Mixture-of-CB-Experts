import torch
import torchvision
import transformers
from torch import nn

class BaseEncoder(nn.Module):
    """
    Base class for encoders. It provides a common interface for all encoders.
    All encoders should inherit from this class and implement the `forward` method.

    Args:
        input_size (int): Number of input features.
        output_size (int): Number of output targets.
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
        self.linear = nn.Linear(input_size, output_size)
        self.activation = getattr(nn, activation)()

    def forward(self, x):
        if self.input_transform is not None:
            x = self.input_transform(x)
        return self.activation(self.linear(x))


class MLPEncoder(BaseEncoder):
    """
    A simple MLP encoder that can be used as a base for concept models.

    Args:
        input_size (int): Number of input features.
        output_size (int): Number of output targets.
        hidden_size (int): Number of hidden units in the MLP.
        activation (str): Activation function to use in the MLP.
    """

    def __init__(self, input_size, output_size, input_transform=None, hidden_size=64,
                 activation='ReLU'):
        super().__init__(input_size, output_size, input_transform)
        if input_transform is not None:
            self.input_transform.flatten = True
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            getattr(nn, activation)(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        if self.input_transform is not None:
            x = self.input_transform(x)
        return self.mlp(x)


class ResNetEncoder(BaseEncoder):
    """
    A simple ResNet-like encoder that can be used as a base for concept models.
    According to the string passed it will create a resnet18, resnet34,
    resnet50, resnet101 or resnet152.
    """

    def __init__(self, input_size, output_size, input_transform=None, resnet_type='resnet18'):
        super().__init__(input_size, output_size, input_transform)
        if resnet_type == 'resnet18':
            from torchvision.models import resnet18
            self.resnet = resnet18(pretrained=True)
        elif resnet_type == 'resnet34':
            from torchvision.models import resnet34
            self.resnet = resnet34(pretrained=True)
        elif resnet_type == 'resnet50':
            from torchvision.models import resnet50
            self.resnet = resnet50(pretrained=True)
        elif resnet_type == 'resnet101':
            from torchvision.models import resnet101
            self.resnet = resnet101(pretrained=True)
        elif resnet_type == 'resnet152':
            from torchvision.models import resnet152
            self.resnet = resnet152(pretrained=True)
        else:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}")

        # Replace the final fully connected layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_size)

        self.transform = ImageTransform()

    def forward(self, x):
        if self.input_transform is not None:
            x = self.input_transform(x)
        return self.resnet(x)


class VitEncoder(BaseEncoder):
    def __init__(self, input_size, output_size, input_transform=None, model_name='google/vit-base-patch32-224-in21k'):
        super().__init__(input_size, output_size, input_transform)
        self.model = transformers.ViTModel.from_pretrained(model_name)
        self.model.eval()
        self.head = nn.Linear(self.model.config.hidden_size, output_size)

    def forward(self, x):
        if callable(self.input_transform):
            x = self.input_transform(x)
        outputs = self.model(x)
        return self.head(outputs.last_hidden_state[:, 0, :])

class ImageTransform(nn.Module):
    """
    A simple image transformation module that can be used to preprocess images
    before passing them to the ResNet encoder.

    Args:
        input_size (int): Size of the input images (assumed square).
    """

    def __init__(self, flatten=False):
        super().__init__()
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
        ])
        self.flatten = flatten

    def forward(self, x):
        x = self.transform(x)
        if self.flatten:
            x = x.view(x.size(0), -1)
        return x


class MNISTTransform(nn.Module):
    """
    A simple transformation module for MNIST data.
    It normalizes the images to have mean 0 and standard deviation 1.
    """

    def __init__(self, flatten=False):
        super().__init__()
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ])
        self.flatten = flatten

    def forward(self, x):
        x = self.transform(x)
        if self.flatten:
            x = x.view(x.size(0), -1) # Flatten the tensor if using a non ImageEncoder
        else:
            x = x.repeat(1, 3, 1, 1) # Convert to 3 channels if using an ImageEncoder
        return x