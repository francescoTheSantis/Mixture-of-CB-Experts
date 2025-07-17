from torch import nn
from src.models.encoders.base import BaseEncoder
from src.models.encoders.transform import ImageTransform

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