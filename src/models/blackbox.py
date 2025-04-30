import torch
import torch.nn as nn

class BlackBox(nn.Module):
    def __init__(self,
                 input_size,
                 output_size=2,
                 activation='ReLU',
                 task = 'classification',
                 ):
        super(BlackBox, self).__init__()
        
        self.has_concepts = False
        hidden_size = input_size * 128
        
        self.sequential = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            getattr(nn, activation)(),
            nn.Linear(hidden_size, hidden_size),
            getattr(nn, activation)(),
            nn.Linear(hidden_size, output_size)
        )

        if task == 'classification':
            if output_size > 1:
                self.task_loss_form = nn.CrossEntropyLoss()
            if output_size == 1:
                self.task_loss_form = nn.BCEWithLogitsLoss()
        elif task == 'regression':
            self.task_loss_form = nn.MSELoss()

    def forward(self, input):
        x = input['x']
        y_hat = self.sequential(x)
        return y_hat, None
    
    def filter_output_for_loss(self, y_output, c_output=None):
        return y_output, c_output
    
    def loss(self, y_hat, y, c_hat_dict=None, c=None):
        y = y.flatten().long()
        # cross entropy
        loss = self.task_loss_form(y_hat.squeeze(), y)
        return loss