import torch
import torch.nn as nn
import torch_concepts.nn as pyc_nn
from src.models.base import BaseModel
import torch.nn.functional as F
from torch_concepts.nn import concept_embedding_mixture

class SymbolicMemoryReasoner(BaseModel):
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
                 mc_approx=10,
                 concept_loss_form=nn.BCELoss(),
                 backbone_latent_size=None,
                 concept_type='binary',
                 equations=None
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
            equations
        )

        self.embedding_size = embedding_size
        self.has_concepts = True
        self.y_names = list(y_names)
        self.weight_reg = weight_reg

        self.mc_approx = mc_approx
        self.memory_size = memory_size

        # We need to use the Concept embedding model to produce both concept predictions and embeddings.
        # Which will allow to intervene on both the linear classifier selection (concept embeddings)
        # and execution (concept predictions).
        self.bottleneck = pyc_nn.ConceptEmbeddingBottleneck(
            backbone_latent_size,
            self.c_names,
            embedding_size,
            activation=nn.Identity()
        )

        selector_input_size = backbone_latent_size
        selector_output_size = memory_size * len(y_names)
        self.classifier_selector = nn.Sequential(
            nn.Linear(selector_input_size, selector_output_size),
        )

        # Memory TODO

    def compute_tau(self, global_step, tau_init=1, tau_min=0.05, decay_rate=0.99):
        # Exponential decay to decrease tau over time
        tau = max(tau_min, tau_init * decay_rate ** global_step)
        return tau

    def forward(self, input):
        latent, c_true, int_idxs = self.encode(input)
        bsz = latent.shape[0]

        c_emb, c_dict = self.bottleneck(
            latent,
            c_true=c_true,
            intervention_idxs=int_idxs,
            intervention_rate=1,
        )
        c_pred = c_dict['c_int']

        c_pred, input_concepts = self._process_concepts(c_pred, c_true, int_idxs)

        # It is necessary to compute again since 
        c_emb = self.bottleneck.linear(latent)
        c_emb = concept_embedding_mixture(c_emb, input_concepts)

        selector_input = latent

        classifier_selector_logits = self.classifier_selector(selector_input)

        # At training time, we sample multiple times (Monte-Carlo approximation)
        # from a categorical distribution.
        # At inference time, only one sample is taken
        if self.training and self.sampling:
            n_samples = self.mc_approx
        else:
            n_samples = 1

        # Reshape the logits to have dimension (bsz, memory_size, n_classes)
        classifier_selector_logits = classifier_selector_logits.view(-1, self.memory_size, len(self.y_names))
        # Save the distribution over the memory to compute the entropy,
        # which allows to evaluate how peaked the distribution is. 
        selection_dist = classifier_selector_logits.view(bsz*len(self.y_names), self.memory_size).clone().detach()
        # Dimension: (bsz, memory_size, n_classes, n_samples)
        classifier_selector_logits = classifier_selector_logits.unsqueeze(-1).expand(-1, -1, -1, n_samples)

        # Compute the temperature for the Gumbel-Softmax distribution
        current_tau = self.compute_tau(self.global_step)

        # Dimension: (bsz, memory_size, n_samples)
        prob_per_classifier = F.gumbel_softmax(classifier_selector_logits, 
                                                        tau=current_tau, 
                                                        hard=True, 
                                                        dim=1)

        # TODO execute equations (so far just known)

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
        y_pred = self.selection_eval(prob_per_classifier, y_per_classifier)
        if self.bias=='global':
            y_pred = y_pred + self.bias_params[None, :, None]

        return y_pred, c_pred, predicted_weights, selection_dist
    
    def linear_equation_eval(self, memory, input_concepts):
        if self.bias == 'local':
            concept_memory = memory[:,:,:-1,:]
            bias_memory = memory[:,:,-1,:].permute(0, 2, 1)
        else:
            concept_memory = memory
            bias_memory = None
        
        y_pred = torch.einsum('bmcy,bc->bym', concept_memory, input_concepts)

        if bias_memory is not None:
            y_pred = y_pred + bias_memory

        return y_pred

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
        
    def loss(self, y_hat, y, c_hat=None, c=None):
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

    def filter_output_for_metrics(self, y_output, c_output=None, predicted_cbm=None, distribution_over_memory=None):
        # Average over the last dimension, which contains the samples
        # form the Monte Carlo approximation.
        y_output = y_output.mean(dim=-1)
        return y_output, c_output
    
    def filter_output_for_loss(self, y_output, c_output=None, predicted_cbm=None, distribution_over_memory=None):
        # This models return the predicted CBM in addition to the usual
        # y and c predictions. The loss function needs only y and c to be computed.
        return y_output, c_output