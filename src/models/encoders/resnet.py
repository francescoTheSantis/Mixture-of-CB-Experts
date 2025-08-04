from torch import nn
from src.models.encoders.base import BaseEncoder

class ResNetEncoder(BaseEncoder):
    """
    A simple ResNet-like encoder that can be used as a base for concept models.
    According to the string passed it will create a resnet18, resnet34,
    resnet50, resnet101 or resnet152.
    """
    def __init__(self, input_size, output_size=None, input_transform=None, type='resnet18'):
        super().__init__(input_size, output_size, input_transform)
        if type == 'resnet18':
            from torchvision.models import resnet18
            self.resnet = resnet18(pretrained=True)
        elif type == 'resnet34':
            from torchvision.models import resnet34
            self.resnet = resnet34(pretrained=True)
        elif type == 'resnet50':
            from torchvision.models import resnet50
            self.resnet = resnet50(pretrained=True)
        elif type == 'resnet101':
            from torchvision.models import resnet101
            self.resnet = resnet101(pretrained=True)
        elif type == 'resnet152':
            from torchvision.models import resnet152
            self.resnet = resnet152(pretrained=True)
        else:
            raise ValueError(f"Unsupported ResNet type: {type}")

        # Remove the final fully connected layer
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        if self.input_transform is not None:
            x = self.input_transform(x)
        x = self.resnet(x)
        x = x.view(x.size(0), -1) # Flatten the output
        return x