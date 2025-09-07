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
# # os.environ["JULIA_NUM_THREADS"] = "8" # Set the number of threads for Julia (used by PySR)
# from pysr import PySRRegressor

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
                 use_memory=True,
                 disjoint_training=False,
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
            disjoint_training
        )

        self.embedding_size = embedding_size
        self.has_concepts = True
        self.y_names = list(y_names)
        self.weight_reg = weight_reg
        self.output_size = output_size
        self.backbone_latent_size = backbone_latent_size
        self.activation = activation

        self.mc_approx = mc_approx
        self.memory_size = memory_size
        self.use_memory = use_memory
        self.selector_model = selector_model

        # We need to use the Concept embedding model to produce both concept predictions and embeddings.
        self.bottleneck = pyc_nn.ConceptEmbeddingBottleneck(
            backbone_latent_size,
            self.c_names,
            embedding_size,
            activation=nn.Identity()
        )

        # Equations handling
        self.equation_learning_strategy = equation_learning_strategy
        self.known_equations = known_equations

        # Handle parameter inconsistencies
        if len(self.known_equations)>1 and not self.use_memory:
            raise ValueError("If multiple equations are provided, memory must be used to select among them.")
        elif len(self.known_equations)==1 and self.use_memory:
            raise ValueError("Only one equation provided, memory will not be used.")
        
        if self.equation_learning_strategy not in ['prior_knowledge', 'kan']:
            raise ValueError(f"Unknown equation learning strategy: {self.equation_learning_strategy}")
        
    ###### Setup methods ######
    def setup_memory(self):
        if self.equation_learning_strategy=='prior_knowledge': 
            if self.known_equations is None:
                raise ValueError("Known equations must be provided when using 'prior_knowledge' strategy.")
            self._prepare_equations(self.known_equations) # We process the known equations to convert them to torch executable modules
            self.memory_size = len(self.known_equations) # Override memory size to match the number of known equations
        elif self.equation_learning_strategy == 'kan':
            kan_params = {
                    'width': [len(self.c_names), len(self.c_names)+1, self.output_size],
                    'grid': 3,
                    'k': 3,
                    'ckpt_path': os.path.join(os.getcwd(), 'kan_ckpt'),
            }
            # Instantiate as many KAN Layers as the memory size
            self.kan_layers = nn.ModuleList()
            for _ in range(self.memory_size):
                kan_layer = KAN(**kan_params)
                self.kan_layers.append(kan_layer)

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
    def setup_kan_grid(self, inputs):
        # Update the grid of all KAN layers based on the provided inputs
        for kan_layer in self.kan_layers:
            kan_layer.update_grid(inputs)

    def _execute_kan(self, prob_per_classifier, input_concepts):
        # Execute all the KAN layers
        eq_outputs = []
        for i, kan_layer in enumerate(self.kan_layers):
            eq_output = kan_layer(input_concepts)
            eq_outputs.append(eq_output)
        # Stack the outputs along the class dimension
        eq_outputs = torch.stack(eq_outputs, dim=1).squeeze() # Shape: (bsz, n_equations)
        # Combine the outputs using the selector probabilities
        y_hat = torch.einsum('bmts,bm->bts', prob_per_classifier, eq_outputs)

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

        # Combine the outputs using the selector probabilities
        y_hat = torch.einsum('bmts,bm->bts', prob_per_classifier, eq_outputs)

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

        self._setup_string_equations()

        explanations = [self.string_equations[idx.item()] for _, idx in enumerate(eq_idx)]
        return explanations

    def _setup_string_equations(self):
        # if the self.string_equations have not been computed yet, compute them
        if not hasattr(self, 'string_equations'):
            if self.equation_learning_strategy=='prior_knowledge':
                equations = self.known_equations
            elif self.equation_learning_strategy=='kan':
                equations = [kan_layer.symbolic_formula()[0] for kan_layer in self.kan_layers]
                # de-standardize if needed
                if self.task == 'regression' and self.scale_target:
                    y_mean = self.scaler.mean_.item()
                    y_std = self.scaler.std_.item()
                    equations = [((eq * y_std + y_mean)) for eq in equations]
            # convert to string
            self.string_equations = [str(eq) for eq in equations]

    ###### Forward and loss methods ######
    def compute_tau(self, global_step, tau_init=2, tau_min=0.05, decay_rate=0.99):
        # Exponential decay to decrease tau over time
        tau = max(tau_min, tau_init * decay_rate ** global_step)
        return tau

    def forward(self, input):
        latent, c_true, int_idxs = self.encode(input)
        bsz = latent.shape[0]

        ## Concept encoder and concept processing block ##
        c_emb, c_dict = self.bottleneck(
            latent,
            c_true=c_true,
            intervention_idxs=int_idxs,
            intervention_rate=1,
        )
        c_hat = c_dict['c_int']

        c_hat, input_concepts = self._process_concepts(c_hat, c_true, int_idxs)

        c_emb = self.bottleneck.linear(latent)
        c_emb = concept_embedding_mixture(c_emb, input_concepts)

        if self.training:
            n_samples = self.mc_approx
        else:
            n_samples = 1

        ## Memory block ##
        if self.use_memory and self.memory_size>1:
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
            prob_per_classifier = torch.ones((bsz, 1, len(self.y_names)), device=latent.device)
            selection_dist = torch.tensor([0.0], device=latent.device)

        ## Equation execution block ##
        y_hat, explanations = self._execute_equations(prob_per_classifier, input_concepts)

        return {
            'y_hat': y_hat,
            'c_hat': c_hat,
            'explanations': explanations,
            'selection_dist': selection_dist
        }

    def loss(self, y_hat, y, c_hat=None, c=None):
        loss = self.concept_based_loss(y_hat, y, c_hat, c)
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