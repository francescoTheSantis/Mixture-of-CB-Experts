import torch
import torch.nn as nn
from src.models.base import BaseModel
import torch_concepts.nn as pyc_nn
import torch.nn.functional as F

class PredictCBM(BaseModel):
    def __init__(self, 
                 input_size, 
                 output_size,
                 c_names,
                 y_names,
                 task, 
                 task_penalty,
                 activation='ReLU',
                 int_prob=0.1,
                 int_idxs=None,
                 noise=None,
                 latent_size=64,
                 c_groups=None,
                 memory_size=7,
                 weight_reg=1e-4,
                 mc_approx=10,
                 hard_concepts=False,
                 use_bias=True,
                 concept_state_weight=False
                 ):

        super().__init__(
                 input_size, 
                 output_size,
                 task,
                 activation,
                 latent_size,
                 c_groups
                 )

        # Parameters in common with the other Concept Embedding based models.
        self.task_penalty = task_penalty
        self.c_names = list(c_names)
        self.int_prob = int_prob
        self.int_idxs = int_idxs
        self.has_concepts = True
        self.noise = noise
        self.concept_loss_form = nn.BCELoss()
        self.hard_concepts = hard_concepts
        self.use_bias = use_bias

        # The concept predictor
        self.bottleneck = pyc_nn.LinearConceptBottleneck(
            self.latent_size,
            self.c_names,
        )

        # Parameters specific to this model
        self.memory_size = memory_size

        # Assign the weight to each state of the concepts if concept_state_weight is True
        self.concept_state_weight = concept_state_weight

        # The memory bank is composed by multiple linear equations 
        self.memory_bank = nn.Parameter(
                torch.randn(memory_size, output_size, len(c_names))
        )
        # If concept_state_weight is True, we add a complementary memory bank
        # that will be used to store the weights associated to the complementary state of the concepts
        if self.concept_state_weight:
            self.complementary_memory_bank = nn.Parameter(
                    torch.randn(memory_size, output_size, len(c_names))
            )            

        if self.use_bias:
            # The global class biases
            self.biases = nn.Parameter(
                torch.randn(output_size)
            )

        # The selctor which produces the logits for the categorical distribution
        # that will be used to sample the linear equations from the memory bank.
        input_dim = latent_size + 2 * len(self.c_names)
        self.selector = nn.Sequential(
            nn.Linear(input_dim, memory_size * output_size)
        )
    
        # Number of samples for the Monte-Carlo approximation
        self.mc_approx = mc_approx

        # Regularization weight to regulate concept sparsity in the memory bank
        self.weight_reg = weight_reg
    
    def forward(self, input):  
        x, c_true, int_idxs = self.encode(input)

        c_pred, _ = self.bottleneck(
            x,
            c_true=c_true,
            intervention_idxs=int_idxs,
            intervention_rate=1.
        )

        # If hard_concepts is True, convert the predicted concepts to binary values
        if self.hard_concepts:
            input_concepts = (c_pred > 0.5).float()
        else:
            input_concepts = c_pred

        # Predict a CBM from the memory bank and classify the samples in the batch
        y_pred, predicted_cbm, selection_dist = self.classify(x, input_concepts)

        return y_pred, c_pred, predicted_cbm, selection_dist

    def compute_tau(self, global_step, tau_init=1, tau_min=0.1, decay_rate=0.99):
        # The temperature is decayed from initial_temp to min_temp over time
        tau = max(tau_min, tau_init * decay_rate ** global_step)
        return tau

    def classify(self, x, c_pred):

        # Prepare the parameters needed for the selector to sample from the memory bank
        current_tau = self.compute_tau(self.global_step)
        bsz = x.shape[0]
        selection_embedding = torch.cat([x, c_pred, 1-c_pred], dim=-1)

        # For each sample, we select the corresponding CBM in the memory bank.
        # This implies computing the logits of the categorical distribution
        # that will be used to sample the CBM.
        # Dimension: (bsz, memory_size, output_size)
        selection = self.selector(selection_embedding).view(bsz, self.memory_size, self.output_size)
        
        # Store a copy of the selection for metrics
        selection_dist = selection.view(bsz * self.output_size, self.memory_size).clone().detach()
        selection = selection.unsqueeze(-1)

        # Add the batch dimension to the memory bank
        # Dimension: (bsz, memory_size, output_size, len(c_names))
        memory_bank = self.memory_bank.unsqueeze(0).expand(bsz, -1, -1, -1)
        if self.concept_state_weight:
            complementary_memory_bank = self.complementary_memory_bank.unsqueeze(0).expand(bsz, -1, -1, -1)
            # Perform a convex combination of the two memory banks according to the predicted concepts
            memory_bank = memory_bank * c_pred.unsqueeze(1).unsqueeze(1) + \
                          complementary_memory_bank * (1 - c_pred.unsqueeze(1).unsqueeze(1))

        # At training time, we sample multiple times (Monte-Carlo approximation)
        # from a categorical distribution.
        # At inference time, only one sample is taken
        if self.training:
           n_samples = self.mc_approx
        else:
           n_samples = 1

        # Dimension: (bsz, memory_size, output_size, n_samples)
        selection = selection.expand(-1, -1, -1, n_samples)

        # Sample from the categorical distribution using the Gumbel-Softmax
        # Dimension: (bsz, memory_size, output_size, n_samples)
        selection = F.gumbel_softmax(selection, 
                                    tau=current_tau, 
                                    hard=False,
                                    dim=1)

        # Select the CBM from the memory bank
        # Dimension: (bsz, output_size, len(c_names), n_samples)
        predicted_cbm = torch.einsum('bmtc,bmts->btcs', memory_bank, selection)

        # Classify the sample by computing the matrix multiplication
        # between the selected CBM and the concept predictions
        # Dimension: (bsz, output_size, n_samples)
        expanded_c_pred = c_pred.unsqueeze(-1).expand(-1, -1, n_samples)
        y_probs = torch.einsum('bcs,btcs->bts', expanded_c_pred, predicted_cbm)

        if self.use_bias:
            # Add the global biases
            y_probs = y_probs + self.biases.unsqueeze(0).unsqueeze(-1).expand(bsz, -1, n_samples)

        return y_probs, predicted_cbm, selection_dist
    
    def sparsity_loss(self):
        # The sparsity loss is computed as the L1 norm of the memory bank
        loss = self.weight_reg * self.memory_bank.norm(p=1)
        return loss

    def loss(self, y_hat, y, c_hat=None, c=None):
        loss = self.concept_based_loss(y_hat, y, c_hat, c)
        loss += self.sparsity_loss()
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
    
