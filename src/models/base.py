import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size,
                 task='classification',
                 activation='ReLU',
                 latent_size=128,
                 dataset=None
                 ):
        super(BaseModel, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.task = task
        self.latent_size = latent_size
        self.dataset = dataset
        
        if dataset in ['mnist_addition']:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # Reduced filters
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Reduced filters
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.AdaptiveAvgPool2d((1, 1)),  # Global pooling
                nn.Flatten(),
                nn.Linear(32, latent_size),  # Reduced input size
                getattr(nn, activation)()
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_size, latent_size),
                getattr(nn, activation)(),
                nn.Linear(latent_size, latent_size),
                getattr(nn, activation)()
            )

        if task == 'classification':
            self.task_loss_form = nn.CrossEntropyLoss()
        elif task == 'regression':
            self.task_loss_form = nn.MSELoss()