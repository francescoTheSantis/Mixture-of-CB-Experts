from torch import nn
from src.models.encoders.base import BaseEncoder
from transformers import AutoModel

class TransformerEncoder(BaseEncoder):
    """
    A simple encoder that utilizes a Hugging Face transformer model.
    The model type is specified by the string passed to the constructor.
    """
    def __init__(self, input_size, output_size=None, input_transform=None, type='bert-base-uncased'):
        super().__init__(input_size, output_size, input_transform)
        self.model_name = type
        self.transformer = AutoModel.from_pretrained(type)
        self.freeze()  # Freeze the transformer layers by default

    # unfreeze only the last layer of the transformer
    def freeze(self):
        for param in self.transformer.parameters():
            param.requires_grad = False
        for param in self.transformer.encoder.layer[-1].parameters():
            param.requires_grad = True

    def forward(self, x):
        if self.input_transform is not None:
            x = self.input_transform(x)
        # Pass input through the Hugging Face transformer model
        outputs = self.transformer(**x)
        hidden_states = outputs.last_hidden_state  # Use the last hidden state
        x = hidden_states[:, 0, :]  # Use the [CLS] token representation
        return x