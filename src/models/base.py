import torch.nn as nn
import torch

class BaseModel(nn.Module):
    """
    Base class for concept models (and blackbox).

    Args:
        input_size (int): Number of input features.
        output_size (int): Number of output targets.
        task (str): Task type, either 'classification' or 'regression'. Default is 'classification'.
        activation (str): Name of the activation function to use in the encoder (e.g., 'ReLU').
        latent_size (int): Size of the latent representation in the encoder. Default is 64.
        c_groups (dict, optional): Dictionary defining concept groups for interventions.

    Attributes:
        encoder (nn.Sequential): Encoder mapping input to latent space.
        task_loss_form (nn.Module): Loss function for the task.
        concept_loss_form (nn.Module): Loss function for concepts.
        task_penalty (float): Weighting factor for the task loss.
        int_idxs (Tensor): Indices of intervened concepts (Default None).
        test_interventions (bool): Whether to apply interventions during testing.
        c_groups (dict): Concept groups for interventions.
        current_epoch (int): Current training epoch.

    Methods:
        encode(input):
            Encodes input data, computes indxes for applying concept interventions and add noise
            to the latent embedding (if specified).
        concept_based_loss(y_hat, y, c_hat=None, c=None):
            Computes the following loss: L_{task}*task_penalty + L_{concepts} .
        get_intervened_concepts_predictions(labels, groups=None):
            Generates a mask for concept interventions based on intervention probability and groups.
    """
    def __init__(self, 
                 input_size, 
                 output_size,
                 task='classification',
                 activation='ReLU',
                 latent_size=64,
                 c_groups=None,
                 ):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.task = task
        self.latent_size = latent_size
        self.int_idxs = None
        self.test_interventions = False
        self.c_groups = c_groups
        self.current_epoch = 0

        self.encoder = nn.Sequential(
            nn.Linear(input_size, latent_size),
            getattr(nn, activation)()
        )

        # Implement MLP as encoder for mnist_addition
        # Flatten the first dimension of x in the forward pass

        if task == 'classification':
            if output_size > 1:
                self.task_loss_form = nn.CrossEntropyLoss()
            else:
                self.task_loss_form = nn.BCEWithLogitsLoss()
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
         
        # Pass the input through the encoder
        x = self.encoder(x)

        if self.training or self.test_interventions:
            # intervene on the concepts according to the int_prob
            int_idxs = self.get_intervened_concepts_predictions(
                c_true,
                groups=self.c_groups
            )
        else:
            int_idxs = torch.zeros_like(c_true)
        int_idxs = int_idxs.bool()
        
        return x, c_true, int_idxs
    
    def concept_based_loss(self, y_hat, y, c_hat=None, c=None):
        if self.task == 'classification':
            if self.output_size > 1:
                y = y.flatten().long()
            else:
                y = y.flatten().float()
        # task loss
        task_loss = 0
        # In case of Monte Carlo sampling
        if y_hat.ndim == 3:
            for i in range(y_hat.shape[-1]):
                task_loss += self.task_loss_form(y_hat[:,:,i].squeeze(), y)
            task_loss /= y_hat.shape[-1]
        else:
            task_loss = self.task_loss_form(y_hat.squeeze(), y)
        # concept loss
        concept_loss = 0
        for i in range(c.shape[1]):
            concept_loss += self.concept_loss_form(c_hat[:,i], c[:,i])
        concept_loss /= c.shape[1]
        # combine the two losses by considering the task penalty regularization
        loss = concept_loss + self.task_penalty * task_loss
        return loss
        
    def get_intervened_concepts_predictions(self, 
                                            labels, 
                                            groups=None):
        '''
        Function to generate a mask for the intervention process.
        The mask is generated based on the probability of intervention 
        and the mismatch between predictions and labels.
        '''
        if groups is not None:
            n_groups = len(groups)
            # Generate a mask of shape Batch x n_groups
            random_mask = torch.rand((labels.shape[0],n_groups), 
                                     dtype=torch.float,
                                     device=labels.device)
            mask = (random_mask < self.int_prob)
            mask = mask.int()
            # Apply group-based intervention
            group_mask = torch.zeros_like(labels, dtype=torch.int, device=labels.device)
            for idx, (_, group) in enumerate(groups.items()):
                group_mask[:, group] = mask[:, idx].unsqueeze(1).expand(-1, len(group))
            mask = group_mask.int()
        else:
            # Generate a probability mask of the same shape
            random_mask = torch.rand_like(labels, dtype=torch.float)
            # Apply probability threshold only on mismatched elements
            mask = (random_mask < self.int_prob)
            mask = mask.int()

        return mask
    
    def filter_output_for_loss(self, y_output, c_output=None):
        """
        Filter the output of the model for loss computation.
        This method can be overridden in subclasses to customize the output filtering.
        """
        return y_output, c_output
    
    def filter_output_for_metrics(self, y_output, c_output=None):
        """
        Filter the output of the model for metrics computation.
        This method can be overridden in subclasses to customize the output filtering.
        """
        return y_output, c_output