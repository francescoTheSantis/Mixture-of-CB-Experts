from matplotlib.pylab import sample
import torch
import torch.nn as nn
import torch_concepts.nn as pyc_nn
from src.models.base import BaseModel
from src.models.encoders.mlp import MLPEncoder
import torch.nn.functional as F
from torch_concepts.nn import concept_embedding_mixture
import re
import sympy
import sympytorch
#import pysr
import numpy as np
import os
from kan import KAN
# Efficient implementation of KAN:
from src.models.efficient_kan.kan import KAN as EfficientKAN

# This is intended to be the list of symbols that the user can understand
# when looking at the equations produced by the model.
USER_DEFINED_SYMBOLS = ['x','x^2','exp','sin','cos'] 

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
                 embedding_size=16,
                 latent_size=128,
                 c_groups=None,
                 memory_size=7,
                 hard_concepts=False,
                 weight_reg=0,
                 encoder=None,
                 mc_approx=10,
                 selector_model='linear',
                 concept_loss_form=nn.BCELoss(),
                 backbone_latent_size=None,
                 concept_type='binary',
                 known_equations=None,
                 equation_learning_strategy=None,
                 disjoint_training=False,
                 decay_rate='cosine',
                 embedding_memory=False,
                 concept_penalty=1.0,
                 regularize=True,
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
        self.output_size = output_size
        self.backbone_latent_size = backbone_latent_size
        self.activation = activation
        self.decay_rate = decay_rate
        self.embedding_memory = embedding_memory
        self.show_explanations = False
        self.equations_for_explanations_ready = False
        self.regularize = regularize

        self.mc_approx = mc_approx
        self.memory_size = memory_size
        self.selector_model = selector_model

        self.bottleneck = pyc_nn.LinearConceptBottleneck(
            backbone_latent_size,
            self.c_names,
            activation=nn.Identity(), # we will later apply a sigmoid if the concept is boolean
        )

        # Equations handling
        self.equation_learning_strategy = equation_learning_strategy
        self.known_equations = known_equations

        if self.equation_learning_strategy not in ['prior_knowledge', 'kan']:
            raise ValueError(f"Unknown equation learning strategy: {self.equation_learning_strategy}")
        
        if self.equation_learning_strategy == 'kan':
            ##### NOTE #####
            # The original KAN implementation is really slow when dealing with high dimensional inputs and outputs.
            # for this reason we the original implmentation for regression tasks with a single output, and we use an efficient
            # implementation for all the other cases.
            # The advantage in using the original KAN implementation is that it allows to extract the symbolic formula learned by the model.

            width = [len(self.c_names), len(self.c_names)+1, self.output_size]

            if self.output_size == 1:
                kan_params = {
                        'width': width, 
                        'grid': 5,
                        'k': 3,
                }
                # Instantiate as many KAN Layers as the memory size
                self.kan_layers = nn.ModuleList()
                for i in range(self.memory_size):
                    kan_params['ckpt_path'] = os.path.join(os.getcwd(), f'kan{i}_ckpt')
                    # generate a random seed in order to have different initializations
                    kan_params['seed'] = np.random.randint(0, 10000)
                    kan_layer = KAN(**kan_params)
                    for param in kan_layer.get_params():
                        param.requires_grad = True
                    if self.regularize:
                        self.lamb = 0.02 # increase for higher sparsity (e.g., 0.1, 0.2, ...)
                        old_save_act, old_symbolic_enabled = kan_layer.disable_symbolic_in_fit(self.lamb)
                        kan_layer.symbolic_enabled = old_symbolic_enabled
                        kan_layer.save_act  = old_save_act
                    self.kan_layers.append(kan_layer)
            else:
                self.kan_layers = nn.ModuleList()
                for i in range(self.memory_size):
                    kan_layers = EfficientKAN(width)
                    for param in kan_layers.parameters():
                        param.requires_grad = True
                    self.kan_layers.append(kan_layers)
        
    ###### Setup methods ######
    def setup_memory(self):
        if self.equation_learning_strategy=='prior_knowledge': 
            if self.known_equations is None:
                raise ValueError("Known equations must be provided when using 'prior_knowledge' strategy.")
            self._prepare_equations(self.known_equations) # We process the known equations to convert them to torch executable modules
            self.memory_size = len(self.known_equations) # Override memory size to match the number of known equations

        if self.selector_model == 'linear':
            self.classifier_selector = nn.Sequential(
                nn.Linear(self.backbone_latent_size,  self.memory_size * len(self.y_names)),
            )
        elif self.selector_model == 'mlp':
            self.classifier_selector = MLPEncoder(
                input_size=self.backbone_latent_size,
                output_size=self.memory_size * len(self.y_names),
                hidden_size=int(self.backbone_latent_size / 3),
                activation=self.activation,
                dropout=0.5
                )
        else:
            raise ValueError(f"Unknown selector model: {self.selector_model}")

    ###### KAN related methods ######
    def setup_kan_grid(self, grid_inputs):  
        grid_inputs = grid_inputs if grid_inputs.ndim > 1 else grid_inputs.unsqueeze(1)
        # Update the grid of all KAN layers based on the provided inputs
        for kan_layer in self.kan_layers:
            kan_layer.to(grid_inputs.device)
            # if the values are in the range [-1, 1], we do not apply the grid update
            if torch.min(grid_inputs) < -1 or torch.max(grid_inputs) > 1:
                kan_layer.update_grid_from_samples(grid_inputs)

    # def prune_kan_layers(self):
    #     kan_layers = nn.ModuleList()
    #     # Prune each KAN layer by removing unnecessary edges and nodes
    #     for kan_layer in self.kan_layers:
    #         pruned_layer = kan_layer.prune()
    #         kan_layers.append(pruned_layer)
    #     self.kan_layers = kan_layers

    def get_learned_equations(self):
        equations = []
        if self.equation_learning_strategy !='kan' or self.output_size!=1:
            raise ValueError("This method is only implemented for KAN with single output.")
        for i, kan_layer in enumerate(self.kan_layers):
            kan_layer.auto_symbolic()
            equations.append(kan_layer.symbolic_formula()[0][0])
        return equations

    def _execute_kan(self, prob_per_classifier, input_concepts):
        # Execute all the KAN layers
        eq_outputs = []
        for i, kan_layer in enumerate(self.kan_layers):
            if self.output_size==1 and self.regularize:
                eq_output = kan_layer(input_concepts, singularity_avoiding=True, y_th=1000)
            else:
                eq_output = kan_layer(input_concepts)
            eq_outputs.append(eq_output)
        # Stack the outputs along the class dimension
        eq_outputs = torch.stack(eq_outputs, dim=1).squeeze() # Shape: (bsz, n_equations)

        if self.task == 'regression' and self.memory_size == 1:
            # Associate the output of the unique KAN to all the classes
            y_hat = eq_outputs.unsqueeze(-1).unsqueeze(-1).expand(-1, len(self.y_names), -1)
        elif self.task == 'regression' and self.memory_size > 1:
            y_hat = torch.einsum('bmts,bm->bts', prob_per_classifier, eq_outputs)
        elif self.task == 'classification' and self.memory_size == 1:
            eq_outputs = eq_outputs.unsqueeze(1)
            y_hat = torch.einsum('bmts,bmt->bts', prob_per_classifier, eq_outputs)
        else:
            y_hat = torch.einsum('bmts,bmt->bts', prob_per_classifier, eq_outputs)

        # Get the explanations (the selected equations)
        if self.training:
            explanations = None
        else:
            explanations = self._get_explanations(prob_per_classifier, y_hat)
        return y_hat, explanations
        
    ###### Equation conversion methods ######
    def _prepare_equations(self, equations):
        self.torch_equations = []
        self.sympy_variables = []
        self.sympy_equations = []

        # define the variables
        variables = [f'c{i}' for i, name in enumerate(self.c_names)]

        self.string_variables = variables

        # Convert the string equations to torch functions
        for eq in equations:
            self._convert_equation_to_torch(eq, variables)

    def _convert_equation_to_torch(self, equation_str, variables):
        if self.equation_learning_strategy != 'prior_knowledge':
            raise NotImplementedError("This method is only implemented for 'prior_knowledge' strategy.")
        # Define the symbols (variables)
        sympy_vars = sympy.symbols(variables)
        # Define the equation in a textual format
        str_exp = equation_str
        # Convert to sympy expression
        sympy_exp = sympy.sympify(str_exp)
        # standardize the equation if self.scale_target
        if self.task == 'regression' and self.scale_target:
            y_mean = self.scaler.mean_.item()
            y_std = self.scaler.std_.item()
            sympy_exp = (sympy_exp - y_mean) / (y_std)
        # Convert the textual equation into an executable PyTorch module
        torch_exp = sympytorch.SymPyModule(expressions=[sympy_exp])
        # Store the torch module, the sympy variables, and the sympy expression 
        self.torch_equations.append(torch_exp)
        self.sympy_variables.append(sympy_vars)
        self.sympy_equations.append(sympy_exp)

    ###### Equation execution methods ######
    def _execute_equations(self, prob_per_classifier, input_concepts):
        bsz = input_concepts.shape[0]

        if self.equation_learning_strategy == 'prior_knowledge':
            y_hat, explanations = self._execute_known_equations(prob_per_classifier, input_concepts)
        elif self.equation_learning_strategy=='kan':
            y_hat, explanations = self._execute_kan(prob_per_classifier, input_concepts)
            
        return y_hat, explanations
    
    def _execute_known_equations(self, prob_per_classifier, input_concepts, discard_explanations=False):
        # Execute all the equations
        eq_outputs = []
        for i, equation in enumerate(self.torch_equations):
            # Create a dictionary mapping variable names to their values
            var_dict = dict(zip(self.string_variables, [input_concepts[:, i] for i in range(input_concepts.shape[1])]))
            # Execute the equation function with the mapped variables
            eq_output = equation(**var_dict)
            eq_outputs.append(eq_output)

        # Stack the outputs along the class dimension
        eq_outputs = torch.stack(eq_outputs, dim=1).squeeze() # Shape: (bsz, n_equations)

        if self.task == 'regression' and self.memory_size == 1:
            # Associate the output of the unique KAN to all the classes
            y_hat = eq_outputs.unsqueeze(-1).unsqueeze(-1).expand(-1, len(self.y_names), -1)
        elif self.task == 'regression' and self.memory_size > 1:
            y_hat = torch.einsum('bmts,bm->bts', prob_per_classifier, eq_outputs)
        elif self.task == 'classification' and self.memory_size == 1:
            eq_outputs = eq_outputs.unsqueeze(1)
            y_hat = torch.einsum('bmts,bmt->bts', prob_per_classifier, eq_outputs)
        else:
            y_hat = torch.einsum('bmts,bmt->bts', prob_per_classifier, eq_outputs)

        if discard_explanations:
            return y_hat
        else:
            # Get the explanations (the selected equations)
            if self.training:
                explanations = None
            else:
                explanations = self._get_explanations(prob_per_classifier, y_hat)
            return y_hat, explanations

    def _get_explanations(self, prob_per_classifier, y_hat):
        # If the number of classes is bigger than one, show the equation selected for the predicted class.
        # Otherwise, show the equation selected for the single output.
        exp_selection = prob_per_classifier[:,:,:,0]
        # select the predicted class
        y_idx = y_hat.argmax(dim=1).squeeze().unsqueeze(1).expand(-1, exp_selection.size(1)).unsqueeze(-1)
        eq_idx = torch.gather(exp_selection, 2, y_idx).squeeze(-1).argmax(1)

        if self.show_explanations:
            if not self.equations_for_explanations_ready:
                self._setup_string_equations()
            explanations = [self.string_equations[idx.item()] for _, idx in enumerate(eq_idx)]
        else:
            explanations = None

        return explanations

    def _setup_string_equations(self):
        # If set to true, this function will never be called again
        self.equations_for_explanations_ready = True

        if self.equation_learning_strategy=='prior_knowledge':
            equations = self.known_equations
        elif self.equation_learning_strategy=='kan':
            equations = []
            #for i, kan_layer in enumerate(self.kan_layers):
                # Plot the kan layer
                #kan_layer.auto_symbolic(lib=USER_DEFINED_SYMBOLS)
                #kan_layer.auto_symbolic()
                #equations.append(kan_layer.symbolic_formula()[0][0])

        # convert to string
        self.string_equations = [str(eq) for eq in equations]

    ###### Forward and loss methods ######
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

        latent, x_concepts, c_true, int_idxs = self.encode(input)
        bsz = latent.shape[0]

        ## Concept encoder and concept processing block ##
        c_hat, _ = self.bottleneck(x_concepts)

        c_hat, input_concepts = self._process_concepts(c_hat, c_true, int_idxs)

        if self.training:
            n_samples = self.mc_approx
        else:
            n_samples = 1

        ## Memory block ##
        if self.memory_size>1:
            classifier_selector_logits = self.classifier_selector(latent)
            # Reshape the logits to have dimension (bsz, memory_size, n_classes)
            classifier_selector_logits = classifier_selector_logits.view(-1, self.memory_size, len(self.y_names))
            # Save the distribution over the memory to compute the entropy,
            # which allows to evaluate how peaked the distribution is. 
            selection_dist = classifier_selector_logits.view(bsz*len(self.y_names), self.memory_size).clone().detach()
            # Dimension: (bsz, memory_size, n_classes, n_samples)
            classifier_selector_logits = classifier_selector_logits.unsqueeze(-1).expand(-1, -1, -1, n_samples)
            # Compute the temperature for the Gumbel-Softmax distribution
            current_tau = self.compute_tau(self.global_step)
            # Dimension: (bsz, memory_size, n_classes, n_samples)
            prob_per_classifier = F.gumbel_softmax(classifier_selector_logits, 
                                                   tau=current_tau, 
                                                   hard=True, 
                                                   dim=1)
        else:
            prob_per_classifier = torch.ones((bsz, 1, len(self.y_names), 1), device=latent.device)
            selection_dist = torch.tensor([0.0], device=latent.device)

        ## Equation execution block ##
        y_hat, explanations = self._execute_equations(prob_per_classifier, input_concepts)

        return {
            'y_hat': y_hat,
            'c_hat': c_hat,
            'explanations': explanations,
            'selection_dist': selection_dist
        }

    def kan_regularization_term(self):
        reg_term = 0.0
        for kan_layer in self.kan_layers:
            reg_term += kan_layer.get_reg(reg_metric="edge_forward_spline_n", lamb_l1=1, lamb_entropy=2, lamb_coef=0, lamb_coefdiff=0)
        return reg_term * self.lamb

    def loss(self, y_hat, y, c_hat=None, c=None):
        loss = self.concept_based_loss(y_hat, y, c_hat, c)
        if self.output_size == 1 and self.regularize and self.equation_learning_strategy=='kan':
            loss += self.kan_regularization_term()
        return loss
    


    # def _fit_symbolic_reg_model(self, top_fraction=0.5):
    #     equations = []
    #     residuals = np.zeros_like(self.y_trues)

    #     X_current, y_current = self.c_trues, self.y_trues

    #     for i in range(self.memory_size):
    #         # Train symbolic regression on current subset
    #         model = PySRRegressor(
    #             niterations=40,
    #             populations=30,
    #             binary_operators=["+", "-", "*", "/"],
    #             unary_operators=["square", "exp", "log"],
    #             model_selection="best",
    #             verbosity=1,
    #         )
    #         model.fit(X_current, y_current)

    #         # Save equation
    #         eq = model.get_best()["equation"]
    #         str_eq = str(eq)
    #         equations.append(str_eq)

    #         # Compute residuals on full dataset
    #         y_pred_full = model.predict(self.c_trues)
    #         residuals = self.y_trues - y_pred_full

    #         # Select hardest samples for next round
    #         errors = np.abs(residuals)
    #         cutoff = np.quantile(errors, 1 - top_fraction)
    #         mask = errors >= cutoff
    #         X_current, y_current = self.c_trues[mask], self.y_trues[mask]

    #     return equations