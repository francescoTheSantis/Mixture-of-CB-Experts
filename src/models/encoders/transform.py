from torch import nn
import torchvision

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


class VitTransform(nn.Module):
    """
    A simple image transformation module that can be used to preprocess images
    before passing them to the ViT encoder.

    Args:
        input_size (int): Size of the input images (assumed square).
    """

    def __init__(self, flatten=False):
        super().__init__()
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                std=[0.5, 0.5, 0.5])
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
            torchvision.transforms.Resize((224, 224))
        ])
        self.flatten = flatten

    def forward(self, x):
        x = self.transform(x)
        if self.flatten:
            x = x.view(x.size(0), -1) # Flatten the tensor if using a non ImageEncoder
        else:
            x = x.repeat(1, 3, 1, 1) # Convert to 3 channels if using an ImageEncoder
        return x