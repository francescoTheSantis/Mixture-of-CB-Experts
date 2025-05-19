import torch.nn as nn
import torch

class BaseModel(nn.Module):
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

    '''
    def get_intervened_concepts_predictions_to_refine(predictions, labels, probability, all_entries=False, groups=None):

        hard_predictions = torch.where(predictions > 0.5, 1, 0) 
            
        # Find mismatched indices if all_entries is False, select all otherwise
        if all_entries:
            mismatched_mask = (torch.ones_like(hard_predictions))#.nonzero(as_tuple=False)
        else:
            mismatched_mask = (hard_predictions != labels)#.nonzero(as_tuple=False)

        # Generate a probability mask of the same shape
        random_mask = torch.rand_like(predictions, dtype=torch.float)

        # Apply probability threshold only on mismatched elements
        mask = (random_mask < probability) & mismatched_mask
        mask = mask.int()

        if groups is not None:
            # Apply group-based intervention
            for name, group in groups.items():
                group_mask = torch.zeros_like(predictions, dtype=torch.int)
                group_mask[:, group] = mask[:, group]
                mask = torch.max(mask, group_mask)

        return mask
    '''