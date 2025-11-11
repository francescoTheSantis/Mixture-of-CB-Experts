import torch
import torch.nn as nn
import torch_concepts.nn as pyc_nn
from models.baselines.base import BaseModel
import torch.nn.functional as F
from torch_concepts.nn import concept_embedding_mixture
from src.models.encoders.mlp import MLPEncoder
import numpy as np
from src.utils.expression_utils import linear_classifier_expression, store_eq

class LinearMemoryReasoner(BaseModel):
    def __init__(self, 
                 output_size,
                 c_names,
                 y_names,
                 task, 
                 task_penalty,
                 activation='ReLU',
                 int_prob=0.1,
                 int_idxs=None,
                 noise=None,
                 embedding_size = 16,
                 latent_size = 128,
                 c_groups=None,
                 memory_size=7,
                 hard_concepts=False,
                 weight_reg=0,
                 encoder=None,
                 mc_approx=1,
                 embedding_memory=True,
                 selector_model='linear',
                 linear_classifier_selection=False,
                 sampling=True,
                 concept_loss_form=nn.BCELoss(),
                 backbone_latent_size=None,
                 concept_type='binary',
                 decay_rate='cosine',
                 bias=None,
                 disjoint_training=False,
                 concept_penalty=1.0,
                 **kwargs
                 ):

        super().__init__(
            output_size,
            c_names,
            y_names,
            task,
            task_penalty,
            hard_concepts,
            activation,
            int_prob,
            int_idxs,
            noise,
            latent_size,
            c_groups,
            encoder,
            backbone_latent_size,
            concept_type,
            disjoint_training,
            concept_penalty
        )

        self.embedding_size = embedding_size
        self.has_concepts = True
        self.y_names = list(y_names)
        self.weight_reg = weight_reg

        self.sampling = sampling
        self.mc_approx = mc_approx
        self.embedding_memory = embedding_memory
        self.memory_size = memory_size
        self.linear_classifier_selection = linear_classifier_selection
        self.selector_model = selector_model
        self.decay_rate = decay_rate

        self.bias = None if bias is None else bias

        if self.bias not in [None, 'local', 'global']:
            raise ValueError("Invalid bias type. Expected one of [None, 'local', 'global'].")

        self.bottleneck = pyc_nn.LinearConceptBottleneck(
            backbone_latent_size,
            self.c_names,
            activation=nn.Identity(), # we will later apply a sigmoid if the concept is boolean
        )

        # The selector generates logits that define a probability distribution 
        # over the linear equations stored in memory.
        # More precisely, for each class in y_names, we have a set of linear equations in the memory, and the selector
        # selects a linear equation for each class in y_names.
        selector_input_size = backbone_latent_size
        selector_output_size = memory_size if self.linear_classifier_selection else memory_size * len(y_names)
        if self.selector_model == 'linear':
            self.classifier_selector = nn.Sequential(
                nn.Linear(selector_input_size, selector_output_size),
            )
        elif self.selector_model == 'mlp':
            self.classifier_selector = MLPEncoder(
                input_size=selector_input_size,
                output_size=selector_output_size,
                hidden_size=selector_input_size,
                activation=activation,
            )
        else:
            raise ValueError(f"Unknown selector model: {self.selector_model}")

        # The memory containing the set linear equations for each class in self.y_names.
        # It can be instantiated in two ways:
        # 1. using embeddings to represent each cell of the memory, and then use a decoder to 
        #    associate a set of parameters (weights) to each embedding.
        # 2. directly learning the classifiers' parameters by using torch.nn.Parameter
        parameters = self.c_names + ['bias'] if self.bias == 'local' else self.c_names
        self.equation_memory = torch.nn.Embedding(
            memory_size,
            latent_size
        )
        self.equation_decoder = pyc_nn.LinearConceptLayer(
            latent_size,
            [
                parameters,
                self.y_names,
            ],
        )

        # Global bias
        if self.bias == 'global':
            self.bias_params = nn.Parameter(torch.zeros(len(self.y_names))) 

    def compute_tau(self, global_step, tau_init=2, tau_min=0.05, decay_rate=0.99):
        if self.decay_rate == 'linear':
            # Linear decay to decrease tau over time
            tau = max(tau_min, tau_init - decay_rate * global_step)
        elif self.decay_rate == 'exp':
            # Exponential decay to decrease tau over time
            tau = max(tau_min, tau_init * decay_rate ** global_step)
        elif self.decay_rate == 'cosine':
            # Cosine decay to decrease tau over time
            tau = tau_min + (tau_init - tau_min) * (1 + np.cos(np.pi * global_step / 10000)) / 2
        else:
            raise ValueError(f"Unknown decay rate: {self.decay_rate}")
        return tau

    def forward(self, input):
        latent, latent_concepts, c_true, int_idxs = self.encode(input)
        bsz = latent.shape[0]

        c_hat, _ = self.bottleneck(latent_concepts)

        c_hat, input_concepts = self._process_concepts(c_hat, c_true, int_idxs)

        classifier_selector_logits = self.classifier_selector(latent)

        # At training time, we sample multiple times (Monte-Carlo approximation)
        # from a categorical distribution.
        # At inference time, only one sample is taken
        if self.training and self.sampling:
            n_samples = self.mc_approx
        else:
            n_samples = 1

        if not self.linear_classifier_selection:
            # Reshape the logits to have dimension (bsz, memory_size, n_classes)
            classifier_selector_logits = classifier_selector_logits.view(-1, self.memory_size, len(self.y_names))
            # Save the distribution over the memory to compute the entropy,
            # which allows to evaluate how peaked the distribution is. 
            selection_dist = classifier_selector_logits.view(bsz*len(self.y_names), self.memory_size).clone().detach()
            # Dimension: (bsz, memory_size, n_classes, n_samples)
            classifier_selector_logits = classifier_selector_logits.unsqueeze(-1).expand(-1, -1, -1, n_samples)
        else:
            selection_dist = classifier_selector_logits.clone().detach()
            # Dimension: (bsz, memory_size, n_samples)
            classifier_selector_logits = classifier_selector_logits.unsqueeze(-1).expand(-1, -1, n_samples)

        if self.sampling:
            # Compute the temperature for the Gumbel-Softmax distribution
            current_tau = self.compute_tau(self.global_step)

            # Dimension: (bsz, memory_size, n_classes, n_samples)
            prob_per_classifier = F.gumbel_softmax(classifier_selector_logits, 
                                                            tau=current_tau, 
                                                            hard=True, 
                                                            dim=1)
        else:
            # Dimension: (bsz, memory_size, n_samples)
            prob_per_classifier = F.softmax(classifier_selector_logits, dim=1)

        # Get the parameters of the linear equations stored in memory.
        equation_weights = self.equation_decoder(self.equation_memory.weight)

        # Adding batch dimension to concept memory
        equation_weights = equation_weights.unsqueeze(dim=0).expand(bsz, -1, -1, -1)

        # Get the weights to generate the explanation
        predicted_weights = self.get_weights_for_explanation(equation_weights, 
                                                             prob_per_classifier)

        # Execute the linear equations stored in memory by performing the dot product 
        # among the input concepts and the weights of the linear equations.
        # Dimension: (batch_size, output_size, memory_size)
        y_per_classifier = self.linear_equation_eval(equation_weights, input_concepts)
        
        # Select one logit for each class of y form the memory
        # Dimension: (batch_size, output_size, n_samples)
        y_hat = self.selection_eval(prob_per_classifier, y_per_classifier)
        if self.bias=='global':
            y_hat = y_hat + self.bias_params[None, :, None]

        return {
            'y_hat': y_hat,
            'c_hat': c_hat,
            'explanations': predicted_weights,
            'selection_dist': selection_dist
        }

    def linear_equation_eval(self, memory, input_concepts):
        if self.bias == 'local':
            concept_memory = memory[:,:,:-1,:]
            bias_memory = memory[:,:,-1,:].permute(0, 2, 1)
        else:
            concept_memory = memory
            bias_memory = None
        
        y_hat = torch.einsum('bmcy,bc->bym', concept_memory, input_concepts)

        if bias_memory is not None:
            y_hat = y_hat + bias_memory

        return y_hat

    def selection_eval(self, prob_per_classifier, y_per_classifier):
        """
        Select the linear classifier from the memory based on the probabilities
        computed by the classifier selector.
        The output dimension is (batch_size, output_size, n_samples).
        """
        if self.linear_classifier_selection:
            return torch.einsum('bms,btm->bts', prob_per_classifier, y_per_classifier)
        else:
            return torch.einsum('bmts,btm->bts', prob_per_classifier, y_per_classifier)
    
    def get_weights_for_explanation(self, memory, selection):
        """
        Get the classifier's weights selection according to the distribution probabilities.
        The output dimension is (batch_size, output_size, n_concepts, n_samples).
        """
        if self.linear_classifier_selection:
            return torch.einsum('bmct,bms->btcs', memory, selection)
        else:
            return torch.einsum('bmct,bmts->btcs', memory, selection)
        
    def loss(self, y_hat, y, c_hat=None, c=None, *args, **kwargs):
        loss = self.concept_based_loss(y_hat, y, c_hat, c)

        # Collect all the parameters in the memory
        params = self.equation_decoder(self.equation_memory.weight)

        if self.bias == 'local':
            weights = params[:,:-1,:]
            bias = params[:,-1,:]
        else:
            weights = params
            bias = None

        # L1 Regularization over weights
        loss += self.weight_reg * weights.abs().sum()
        # L2 regularization over the bias (if used)
        if self.bias == 'local':
            loss += self.weight_reg * bias.pow(2).sum()
        elif self.bias == 'global':
            loss += self.weight_reg * self.bias_params.pow(2).sum()

        return loss

    def get_symbolic_equivalent(self, log_dir=None):
        """
        Returns the equation associated to the predictor of the model
        """
        
        # Return the most complex linear equation that can obtained after training (all concepts are relevant) 
        bias = True if self.bias != None else False
        equation = linear_classifier_expression(len(self.c_names), include_bias=bias)
        store_eq(equation, log_dir)