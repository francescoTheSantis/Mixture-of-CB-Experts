import torch.nn as nn
import torch
class BaseModel(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size,
                 task='classification',
                 activation='ReLU',
                 latent_size=64,
                 dataset=None
                 ):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.task = task
        self.latent_size = latent_size
        self.dataset = dataset
        
        if dataset in ['mnist_addition']:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), 
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.AdaptiveAvgPool2d((1, 1)),  
                nn.Flatten(),
                nn.Linear(32, latent_size), 
                getattr(nn, activation)()
            )
        elif dataset in ['checkmark', 'xor', 'dot', 'trigonometry']:
            self.encoder = nn.Sequential(
                nn.Linear(input_size, latent_size),
                getattr(nn, activation)()
            )

        if task == 'classification':
            self.task_loss_form = nn.CrossEntropyLoss()
        elif task == 'regression':
            self.task_loss_form = nn.MSELoss()

        self.concept_loss_form = None
        self.task_penalty = None

    def encode(self, input):
        x = input['x']
        c_true = input['c']
    
        # If noise is provided, create a convex combination of the input and noise
        if self.noise!=None:
            eps = torch.randn_like(x)
            x = eps * self.noise + x * (1-self.noise)
            
        x = self.encoder(x)

        # If the intervention index is not provided, 
        # all concept can be selected for interventions
        int_idxs = self.int_idxs if self.int_idxs is not None \
            else torch.ones_like(c_true).bool()
        return x, c_true, int_idxs
    
    def concept_based_loss(self, y_hat, y, c_hat=None, c=None):
        # task loss
        task_loss = self.task_loss_form(y_hat.squeeze(), y)
        # concept loss
        concept_loss = 0
        for i in range(c.shape[1]):
            concept_loss += self.concept_loss_form(c_hat[:,i], c[:,i])
        concept_loss /= c.shape[1]
        # combine the two losses
        loss = concept_loss + self.task_penalty * task_loss
        return loss