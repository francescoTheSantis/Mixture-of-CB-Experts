import torch
import torch.nn as nn
import torch_concepts.nn as pyc_nn
from src.models.base import BaseModel
import torch.nn.functional as F

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
                 negative_concepts=False,
                 hard_concepts=False,
                 weight_reg=1e-4,
                 encoder=None,
                 mc_approx=10,
                 embedding_memory=True,
                 intervene_on_selection=True,
                 linear_classifier_selection=False,
                 cos_sim=True,
                 sampling=True
                 ):

        super().__init__(
            output_size,
            task,
            activation,
            latent_size,
            c_groups,
            encoder
        )

        # Parameters in common with other Concept Embedding-based Models
        self.embedding_size = embedding_size
        self.latent_size = latent_size
        self.task_penalty = task_penalty
        self.c_names = list(c_names)
        self.int_prob = int_prob
        self.int_idxs = int_idxs
        self.has_concepts = True
        self.noise = noise
        self.y_names = list(y_names)
        self.negative_concepts = negative_concepts
        self.hard_concepts = hard_concepts
        self.weight_reg = weight_reg

        # Parameters specific for the LinearMemoryReasoner
        self.sampling = sampling
        self.mc_approx = mc_approx
        self.embedding_memory = embedding_memory
        self.memory_size = memory_size
        self.intervene_on_selection = intervene_on_selection
        self.linear_classifier_selection = linear_classifier_selection
        self.cos_sim = cos_sim

        # If the user, with interventions, wants to modify both the selection of the linear classifier
        # and the execution of the linear classifier, we need to use the 
        # Concept embedding model to produce both concept predictions and embeddings.
        # Which will allow to interven on both the linear classifier selection (concept embeddings) 
        # and execution (concept predictions).
        if self.intervene_on_selection:
            self.bottleneck = pyc_nn.ConceptEmbeddingBottleneck(
                latent_size,
                self.c_names,
                embedding_size,
            )
        else:
            # If the user wants to intervene only on the execution of the linear classifier,
            # we can use the simpler LinearConceptBottleneck, which only produces concept predictions.
            self.bottleneck = pyc_nn.LinearConceptBottleneck(
                self.latent_size,
                self.c_names,
            )            

        # The selector generates logits that define a probability distribution 
        # over the linear equations stored in memory.
        # More precisely, for each class in y_names, we have a set of linear equations in the memory, and the selector
        # selects a linear equation for each class in y_names.
        selector_input_size = embedding_size * len(c_names) if self.intervene_on_selection else latent_size
        if self.linear_classifier_selection:
            self.classifier_selector = nn.Sequential(
                nn.Linear(selector_input_size, memory_size),
            )
        else:
            self.classifier_selector = nn.Sequential(
                nn.Linear(selector_input_size, memory_size * len(y_names)),
            )

        # The memory containing the set linear equations for each class in self.y_names.
        # It can be instantiated in two ways:
        # 1. using embeddings to represent each cell of the memory, and then use a decoder to 
        #    associate a set of parameters (weights) to each embedding.
        # 2. directly learning the classifiers' parameters by using torch.nn.Parameter
        if self.embedding_memory:
            self.equation_memory = torch.nn.Embedding(
                memory_size,
                latent_size
            )
            self.equation_decoder = pyc_nn.LinearConceptLayer(
                latent_size,
                [
                    self.c_names,
                    self.y_names,
                ],
            )
        else: 
            self.equation_memory = nn.Parameter(
                torch.randn(memory_size, len(c_names), output_size)
            )

        self.concept_loss_form = nn.BCELoss()

        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.softplus = nn.Softplus()

    def compute_tau(self, global_step, tau_init=1, tau_min=0.1, decay_rate=0.99):
        # The temperature is decayed from initial_temp to min_temp over time
        tau = max(tau_min, tau_init * decay_rate ** global_step)
        return tau
    
    def transform_concepts(self, c_pred):
        if self.hard_concepts:
            input_concepts = (c_pred > 0.5).float()
        else:
            input_concepts = c_pred
        if self.negative_concepts:
            input_concepts = 2*input_concepts - 1 
        else:
            input_concepts = input_concepts
        return input_concepts

    def forward(self, input):
        latent, c_true, int_idxs = self.encode(input)
        bsz = latent.shape[0]

        if self.intervene_on_selection:
            c_emb, c_dict = self.bottleneck(
                latent,
                c_true=c_true,
                intervention_idxs=int_idxs,
                intervention_rate=1,
            )
            c_pred = c_dict['c_int']
            selector_input = c_emb.flatten(-2)
        else:
            c_pred, _ = self.bottleneck(
                latent,
                c_true=c_true,
                intervention_idxs=int_idxs,
                intervention_rate=1.,
            )
            selector_input = latent

        classifier_selector_logits = self.classifier_selector(selector_input)

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

            # Dimension: (bsz, memory_size, n_samples)
            prob_per_classifier = F.gumbel_softmax(classifier_selector_logits, 
                                                            tau=current_tau, 
                                                            hard=False, 
                                                            dim=1)
        else:
            # Dimension: (bsz, memory_size, n_samples)
            prob_per_classifier = F.softmax(classifier_selector_logits, dim=1)

        # Get the parameters of the linear equations stored in memory.
        if self.embedding_memory:
            equation_weights = self.equation_decoder(self.equation_memory.weight)
        else:
            equation_weights = self.equation_memory
        # Adding batch dimension to concept memory
        equation_weights = equation_weights.unsqueeze(dim=0).expand(bsz, -1, -1, -1)

        input_concepts = self.transform_concepts(c_pred)

        # Get the weights to generate the explanation
        predicted_weights = self.get_weights_for_explanation(equation_weights, prob_per_classifier)

        # Execute the linear equations stored in memory by performing the dot product 
        # among the input concepts and the weights of the linear equations.
        # Dimension: (batch_size, output_size, memory_size)
        y_per_classifier = self.linear_equation_eval(equation_weights, 
                                                   input_concepts)
        
        # Select one logit for each class of y form the memory
        # Dimension: (batch_size, output_size, n_samples)
        y_pred = self.selection_eval(prob_per_classifier, y_per_classifier)

        return y_pred, c_pred, predicted_weights, selection_dist
    
    def linear_equation_eval(self, memory, input_concepts):
        if self.cos_sim:
            # Normalize over the memory dimension
            memory = F.normalize(memory, p=2, dim=2)
            input_concepts = F.normalize(input_concepts, p=2, dim=1)
        y_pred = torch.einsum('bmcy,bc->bym', memory, input_concepts)
        if self.cos_sim:
            y_pred = self.softplus(self.scale) * y_pred
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
        # Add L1 regularization on the weights of the equation memory
        # to encourage sparsity
        if self.embedding_memory:
            loss += self.weight_reg * self.equation_decoder(self.equation_memory.weight).norm(p=1)
        else:
            loss += self.weight_reg * self.equation_memory.norm(p=1)
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
