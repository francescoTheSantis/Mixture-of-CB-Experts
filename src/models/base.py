import torch.nn as nn
import torch
import torch.nn.functional as F
from src.models.encoders.base import BaseEncoder
import copy

class BaseModel(nn.Module):
    """
    Base class for concept models (and blackbox).
    """
    def __init__(self, 
                 output_size,
                 c_names,
                 y_names,
                 task,
                 task_penalty,
                 hard_concepts,
                 activation='ReLU',
                 int_prob=0.1,
                 int_idxs=None,
                 noise=None,
                 latent_size=64,
                 c_groups=None,
                 encoder: BaseEncoder=None,
                 backbone_latent_size=None,
                 concept_type='binary',
                 disjoint_training=False,
                 concept_penalty=1.0
                 ):
        super().__init__()
        
        self.output_size = output_size
        self.task = task
        self.latent_size = latent_size
        self.int_idxs = None
        self.test_interventions = False
        self.c_groups = c_groups
        self.global_step = 0
        self.encoder = encoder
        self.concept_type = concept_type
        self.hard_concepts = hard_concepts
        self.task_penalty = task_penalty
        self.c_names = list(c_names)
        self.int_prob = int_prob
        self.int_idxs = int_idxs
        self.has_concepts = None # This value has to be overriden by the inheriting class
        self.noise = noise
        self.disjoint_training = disjoint_training
        self.concept_penalty = concept_penalty

        self.logic_reasoning = False # This value has to be overriden by the inheriting class if it is a logic-based model

        if task == 'classification':
            if output_size > 1:
                self.task_loss_form = nn.CrossEntropyLoss()
            else:
                self.task_loss_form = nn.BCEWithLogitsLoss()
        elif task == 'regression':
            self.task_loss_form = nn.MSELoss()
        elif task == 'generation':
            self.task_loss_form = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"Task {task} is not implemented. "
                                      f"Supported tasks are 'classification', "
                                      f"'regression', and 'generation'.")

        # The concept loss form is a list of losses. 
        # Each loss in the list is specifically selected according to the concept type.
        self.concept_loss_form = []
        for i in concept_type:
            if i == 'binary':
                self.concept_loss_form.append(nn.BCELoss())
            else:
                self.concept_loss_form.append(nn.MSELoss())

        # Instantiate another encoder for the concepts. This encoder is a copy of the main encoder.
        # It will be used to encode the concepts when disjoint training is enabled.
        if self.disjoint_training:
            self.concept_encoder = copy.deepcopy(encoder)
            # Freeze the weights of the concept encoder
            for param in self.concept_encoder.parameters():
                param.requires_grad = True
                
    def encode(self, input):
        x = input['x']
        c_true = input['c']
         
        # Pass the input through the encoder
        h = self.encoder(x)

        # encode the concepts using the concept encoder
        if self.disjoint_training:
            h_concepts = self.concept_encoder(x)
        else:
            h_concepts = h
        
        # If noise is provided, create a convex combination of the input and noise
        if self.noise!=None:
            eps = torch.randn_like(h)
            h = eps * self.noise + h * (1-self.noise)
            h_concepts = eps * self.noise + h_concepts * (1-self.noise)
            del eps
            
        if self.training or self.test_interventions:
            # intervene on the concepts according to the int_prob
            int_idxs = self.get_intervened_concepts_predictions(
                c_true,
                groups=self.c_groups
            )
        else:
            int_idxs = torch.zeros_like(c_true)
        int_idxs = int_idxs.bool()

        # Maintain the same 2d format even if only one concept is provided
        int_idxs = int_idxs.unsqueeze(-1) if int_idxs.ndim == 1 else int_idxs

        return h, h_concepts, c_true, int_idxs
    
    def _logic_model_checker(self):
        """
        Check if the model is a logic-based model.
        Logic-based models are identified by their class name.
        """
        return self.__class__.__name__ in ['DeepConceptReasoner', 'ConceptMemoryReasoner']

    def _task_loss_variable_check(self, y, y_hat):
        """
        Check the type and shape of y and y_hat before computing the task loss.
        This is useful to ensure that the task loss function receives the correct input format.
        """

        # Check if the model is a logic-based model
        logic_model_check = self._logic_model_checker()
        # Check y type and shape before task loss computation
        if self.task == 'classification':
            if logic_model_check and self.output_size > 1:
                y = F.one_hot(y.flatten().long(), num_classes=self.output_size).float()
            elif self.output_size > 1:
                y = y.flatten().long()
            else:
                y = y.flatten().float()
        elif self.task == 'regression':
            pass
        elif self.task == 'generation':
            # in case of generation, we assume y is a sequence of tokens
            y = y.flatten().long()
        else:
            raise ValueError(f"Unknown task type: {self.task}. Supported tasks are 'classification', 'regression', and 'generation'.")
        return y, y_hat
    
    def _handle_hard_concepts(self, c_hat, int_idxs):
        """
        When the hard_concepts variable is True:
            - the boolean concepts are made hard by applying a threshold at 0.5
            - the integer concepts are made hard by rounding to the nearest integer
            - the floating concepts are left unchanged
        In the locations identified by int_idxs we apply the identity function, as the intervention already happened, 
        and the values do not need to undergo any further transformation.
        """
        if self.hard_concepts:
            # Boolean concepts
            binary_mask = torch.tensor([c_type == 'binary' for c_type in self.concept_type], device=c_hat.device)
            binary_mask = binary_mask & ~int_idxs
            c_hat = torch.where(binary_mask, (c_hat > 0.5).float(), c_hat) if binary_mask.any() else c_hat

            # Integer concepts
            # integer_mask = torch.tensor([c_type == 'integer' for c_type in self.concept_type], device=c_hat.device)
            # integer_mask = integer_mask & ~int_idxs
            # c_hat = torch.where(integer_mask, c_hat.round(), c_hat) if integer_mask.any() else c_hat
        return c_hat

    def _apply_concept_activation(self, c_hat, int_idxs):
        """
        Apply the correct activation function to the concepts:
            - if the concept is boolean, then a bce will be used as loss. For this reason, we apply a sigmoid activation.
            - if the concept is numeric (e.g., integer or floating), then an mse will be used as loss. 
              For this reason, we apply an identity function.
        In the locations identified by int_idxs we apply the identity function, as the intervention already happened, 
        and the values do not need to undergo any further transformation.
        """
        # Create masks for different concept types
        binary_mask = torch.tensor([c_type == 'binary' for c_type in self.concept_type], device=c_hat.device)

        # Combine binary_mask with int_mask
        binary_mask = binary_mask & ~int_idxs
        binary_mask = binary_mask if binary_mask.ndim > 1 else binary_mask.unsqueeze(-1)

        # Apply activations using masks
        c_hat = torch.where(binary_mask, torch.sigmoid(c_hat), c_hat)

        # numeric_mask = ~binary_mask
        # c_hat = torch.where(numeric_mask, c_hat, c_hat)
        return c_hat

    def _process_concepts(self, c_hat, c_true, int_idxs):
        """
        Process the concepts by applying activation, intervening, and handling hard concepts.
        """
        # apply activation to concept prediction
        c_hat = self._apply_concept_activation(c_hat, int_idxs)

        # intervene
        c_hat = self._intervene(c_hat, c_true, int_idxs)

        # Check whether disjoint training is enabled
        if self.disjoint_training and self.training:
            input_concepts = c_true
        else:
            # switch to hard concepts if the corresponding variable is true
            input_concepts = self._handle_hard_concepts(c_hat, int_idxs)

        return c_hat, input_concepts

    def concept_based_loss(self, y_hat, y, c_hat=None, c=None):

        # Update type and shape of y and y_hat before task loss computation
        y, y_hat = self._task_loss_variable_check(y, y_hat)

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
            c_i_loss_form = self.concept_loss_form[i]
            if isinstance(c_i_loss_form, nn.BCELoss) or isinstance(c_i_loss_form, nn.MSELoss):
                concept_loss += c_i_loss_form(c_hat[:,i], c[:,i])
            elif isinstance(c_i_loss_form, nn.CrossEntropyLoss):
                concept_loss = c_i_loss_form(c_hat, c.argmax(-1))
            else:
                raise NotImplementedError(f"{c_i_loss_form} not supported")
        # normalize over the number of concepts to avoid high concept loss
        concept_loss /= c.shape[1]

        # combine the two losses by considering the task & concept penalty regularization
        loss = self.concept_penalty * concept_loss + self.task_penalty * task_loss
        return loss

    def get_intervened_concepts_predictions(self, labels, groups=None):
        """
        Generate the random mask to compute interventions.
        Specifically, we randomly select rows in the batch whose concepts
        will be replaced with their respective ground-truth values.
        """
        bsz = labels.shape[0]
        n_concepts = 1 if labels.ndim==1 else labels.shape[1]

        return (torch.rand(bsz, 1, device=labels.device) < self.int_prob).expand(bsz, n_concepts).int()

    def _intervene(self, c_hat, c_true, int_idxs):
        """
        Apply interventions: when the entry in int_idxs is 1, replace c_hat with c_true
        """
        c_hat = torch.where(int_idxs == 1, c_true, c_hat)
        return c_hat

    def filter_output_for_loss(self, y_hat, c_hat=None, *args, **kwargs):
        """
        Filter the output of the model for loss computation.
        This method can be overridden in subclasses to customize the output filtering.
        """
        return y_hat, c_hat

    def filter_output_for_metrics(self, y_hat, c_hat=None, *args, **kwargs):
        """
        Filter the output of the model for metrics computation.
        This method can be overridden in subclasses to customize the output filtering.
        """
        # If y_hat has 3 dimensions, it means that Monte Carlo sampling is used.
        if y_hat.ndim == 3:
            # Average over the last dimension, which contains the samples
            # form the Monte Carlo approximation.
            y_hat = y_hat.mean(dim=-1)

        # task
        if self.task == 'regression':
            # output will be shape (batch_size, )
            y_hat = y_hat.squeeze()
        elif self.task == 'classification': 
            # output will be shape (batch_size, ) if binary classification
            # output will be shape (batch_size, ) if multi-class classification
            binary_classification = (y_hat.shape[1] == 1)
            if binary_classification:
                # logic-based models return probabilities
                # the other models return logits
                if self.logic_reasoning:
                    pass
                else:
                    y_hat = torch.sigmoid(y_hat)
                y_hat = (y_hat > 0.5).squeeze().long()
            else:
                y_hat = torch.argmax(y_hat, dim=1)
        else:
            raise NotImplementedError(f"Task {self.task} not implemented for metrics computation.")

        # Filter concepts
        if self.has_concepts:
            if self.task == 'regression':
                # output will be shape (batch_size, num_concepts)
                pass
            elif self.task == 'classification':
                # output will be shape (batch_size, num_concepts)
                c_hat = torch.where(c_hat > 0.5, 1, 0)

        return y_hat, c_hat

    def get_symbolic_equivalent(self, log_dir=None):
        """
        Returns the equation associated to the predictor of the model.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_symbolic_equivalent() method"
        )

    # def get_intervened_concepts_predictions(self, labels, groups=None):
    #     '''
    #     Function to generate a mask for the intervention process.
    #     The mask is generated based on the probability of intervention.
    #     '''
    #     with torch.no_grad():
    #         if groups is not None:
    #             # Create the final mask directly without intermediate tensors
    #             mask = torch.zeros_like(labels, dtype=torch.int32, device=labels.device)
    #             batch_size = labels.shape[0]

    #             # Process each group independently to avoid large intermediate tensors
    #             for group_name, group_indices in groups.items():
    #                 # Generate random values only for this group
    #                 random_val = torch.rand(batch_size, device=labels.device, dtype=torch.float32)
    #                 group_mask = (random_val < self.int_prob).int()
                    
    #                 # Apply mask directly to the group indices
    #                 mask[:, group_indices] = group_mask.unsqueeze(1)
                    
    #                 # Clean up immediately
    #                 del random_val, group_mask
                
    #             # Force GPU cache cleanup
    #             if labels.device.type == 'cuda':
    #                 torch.cuda.empty_cache()
                
    #             return mask
    #         else:
    #             # Generate mask directly for non-grouped case
    #             random_values = torch.rand_like(labels, dtype=torch.float32)
    #             mask = (random_values < self.int_prob).int()
                
    #             # Clean up
    #             del random_values
    #             if labels.device.type == 'cuda':
    #                 torch.cuda.empty_cache()
                
    #             return mask
