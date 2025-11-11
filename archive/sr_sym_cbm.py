import torch.nn as nn
import torch_concepts.nn as pyc_nn
from src.models.baselines.base import BaseModel
from src.models.modules.selector import SelectorModel
from src.models.modules.symbolic_predictor import SymbolicPredictor
import sympy as sp
from src.utils.expression_utils import store_eq
import pysr # Ensure pysr is imported
from pysr import PySRRegressor
import numpy as np

binary_operators = ["*", "+", "-", "/"]
unary_operators = ["sin", "cos", "exp", "log", "tan", "tanh"]
# Added in order to have the same functions of the kan model
extra_functions = {
    "inv": lambda x: 1 / x,
    "square": lambda x: x**2,
    "cube": lambda x: x**3,
    "x^4": lambda x: x**4,
    "x^5": lambda x: x**5,
    "inv2": lambda x: 1 / x**2,
    "inv3": lambda x: 1 / x**3,
    "inv4": lambda x: 1 / x**4,
    "inv5": lambda x: 1 / x**5,
    "sqrt": lambda x: x**0.5,
    "x^1.5": lambda x: x**1.5,
    "invsqrt": lambda x: 1 / x**0.5,
    "abs": lambda x: abs(x),
    "sgn": lambda x: 1 if x > 0 else (-1 if x < 0 else 0),
}

class SymbolicRegressorCBM(BaseModel):
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
                 memory_size=1,
                 embedding_size=16,
                 latent_size=128,
                 c_groups=None,
                 hard_concepts=False,
                 encoder=None,
                 mc_approx=1,
                 selector_model='linear',
                 backbone_latent_size=None,
                 concept_type='binary',
                 disjoint_training=False,
                 decay_rate='cosine',
                 embedding_memory=False,
                 concept_penalty=1.0,
                 device='cpu',
                 modality='sr_only',
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
        self.output_size = output_size
        self.backbone_latent_size = backbone_latent_size
        self.activation = activation
        self.embedding_memory = embedding_memory
        self.show_explanations = False
        self.equations_for_explanations_ready = False
        self.device = device
        self.mc_approx = mc_approx
        self.memory_size = memory_size
        self.modality = modality

        # hybrid: first mlps are fit and then symbolic regression is applied on top of them
        # sr_only: only symbolic regression is used to map concepts to targets and the obtained equations are used at inference time
        if self.modality not in ['hybrid', 'sr_only']:
            raise ValueError(f"Modality {self.modality} not recognized. Choose either 'hybrid' or 'sr_only'.")

        # Instantiate the selector
        self.classifier_selector = SelectorModel(
            input_size=self.backbone_latent_size,
            output_size=self.memory_size,
            model_type=selector_model,
            activation=activation,
            decay_rate=decay_rate,
        )
        
        self.bottleneck = pyc_nn.LinearConceptBottleneck(
            backbone_latent_size,
            self.c_names,
            activation=nn.Identity(), # we will later apply a sigmoid if the concept is boolean
        )

        # Store PySR parameters
        self.pysr_params = {
            'populations': 31,
            'population_size': 50,
            'niterations': 100,
            'ncycles_per_iteration': 380, 
            'binary_operators': binary_operators,
            'unary_operators': unary_operators,
            'extra_functions': extra_functions,
            'elementwise_loss': "loss(prediction, target) = (prediction - target)^2",
            'complexity_of_constants': 1,
            'timeout_in_seconds': 10 # 60 * 2, # Stop after 2 minutes
        }

        if self.modality == 'hybrid':
            pass
        elif self.modality == 'sr_only':
            print("Using SR only modality.")
            # Predictor will be initialized after running symbolic regression
            self.predictor = None

    def run_sr(self, c_trues, y_trues):
        """
        Run symbolic regression on the provided concept-target pairs to extract equations.
        
        Args:
            c_trues: Tensor of concept values, shape (n_samples, n_concepts)
            y_trues: Tensor of target values, shape (n_samples, n_targets)
        """
        
        print("\n" + "="*70)
        print("Running Symbolic Regression to extract equations")
        print("="*70)
        
        # Convert tensors to numpy arrays
        X = c_trues.detach().cpu().numpy()  # Shape: (n_samples, n_concepts)
        y = y_trues.detach().cpu().numpy()  # Shape: (n_samples, n_targets)
        
        # Determine number of targets
        n_targets = y.max()
        
        # Dictionary to store equations for each memory slot
        # Structure: {memory_idx: {target_name: sympy_equation}}
        # Each memory slot contains equations for ALL targets
        # Memory slot 0: best equations for all targets
        # Memory slot 1: second best equations for all targets, etc.
        all_equations = {i: {} for i in range(self.memory_size)}
        
        # Dictionary to store top equations for each target
        target_equations = {}
        
        # Dictionary to store ALL equations from the Pareto front for each target
        # This will be used by get_symbolic_equivalent to save all discovered equations
        self.all_pareto_equations = {}
        
        # Run symbolic regression for each target
        for target_idx in range(n_targets):
            target_name = self.y_names[target_idx] if target_idx < len(self.y_names) else f"y_{target_idx}"
            # select the positions where y == target_idx
            y_idx = (y == target_idx).astype(float)
            y_target = y[y_idx.flatten() == 1]
            # Now filter the x accordingly
            X_target = X[y_idx.flatten() == 1]

            print(f"\n--- Processing target: {target_name} ---")
            
            # Initialize PySR model with the specified parameters
            model = PySRRegressor(
                **self.pysr_params,
                verbosity=1,
            )
            
            # Fit the model
            print(f"Fitting PySR model...")
            model.fit(X, y_target)
            
            # Get the Pareto front equations
            equations_df = model.equations_
            print(f"\nPareto front has {len(equations_df)} equations")
            
            # Store ALL equations from Pareto front for this target
            self.all_pareto_equations[target_name] = []
            for idx, row in enumerate(equations_df.itertuples()):
                # Get the sympy equation
                sympy_eq = row.sympy_format
                
                # Rename variables from x0, x1, ... to concept names
                for i, c_name in enumerate(self.c_names):
                    sympy_eq = sympy_eq.subs(sp.Symbol(f'x{i}'), sp.Symbol(c_name))
                
                # Store complete equation info
                self.all_pareto_equations[target_name].append({
                    'equation': sympy_eq,
                    'loss': row.loss,
                    'complexity': row.complexity,
                    'score': row.score
                })
            
            # Select the top memory_size equations based on score
            # Score balances accuracy and complexity
            top_equations = equations_df.nlargest(self.memory_size, 'score')
            
            # Store top equations for this target
            target_equations[target_name] = []
            
            for memory_idx, row in enumerate(top_equations.itertuples()):
                # Get the sympy equation
                sympy_eq = row.sympy_format
                
                # Rename variables from x0, x1, ... to concept names
                for i, c_name in enumerate(self.c_names):
                    sympy_eq = sympy_eq.subs(sp.Symbol(f'x{i}'), sp.Symbol(c_name))
                
                # Store equation info for this target
                target_equations[target_name].append({
                    'equation': sympy_eq,
                    'loss': row.loss,
                    'complexity': row.complexity,
                    'score': row.score
                })
                
                print(f"  Rank {memory_idx}: {sympy_eq}")
                print(f"    Loss: {row.loss:.6f}, Complexity: {row.complexity}, Score: {row.score:.6f}")
        
        # Now organize equations by memory slot
        # Each memory slot gets one equation per target (all with same rank)
        print(f"\n--- Organizing equations into memory slots ---")
        for memory_idx in range(self.memory_size):
            for target_name in target_equations:
                if memory_idx < len(target_equations[target_name]):
                    all_equations[memory_idx][target_name] = target_equations[target_name][memory_idx]['equation']
                    print(f"Memory slot {memory_idx}, {target_name}: {all_equations[memory_idx][target_name]}")
                else:
                    # If we don't have enough equations for this target, use the last available one
                    last_idx = len(target_equations[target_name]) - 1
                    all_equations[memory_idx][target_name] = target_equations[target_name][last_idx]['equation']
                    print(f"Memory slot {memory_idx}, {target_name}: {all_equations[memory_idx][target_name]} (reusing last)")
        
        
        # Instantiate the SymbolicPredictor with the extracted equations
        print(f"\n--- Creating SymbolicPredictor with {len(all_equations)} equation sets ---")
        
        self.predictor = SymbolicPredictor(
            equations=all_equations,
            c_names=self.c_names,
        )
        
        # Move predictor to the correct device
        self.predictor = self.predictor.to(self.device)
        
        print("="*70)
        print("Symbolic Regression completed successfully!")
        print("="*70 + "\n")

    def reset_stored_tensors(self):
        self.stored_inputs = []
        self.stored_outputs = []
        self.memory_idxs = []

    ###### Forward and loss methods ######
    def forward(self, input, store_tensors=False, *args, **kwargs):

        latent, x_concepts, c_true, int_idxs = self.encode(input)

        ## Concept encoder and concept processing block ##
        c_hat, _ = self.bottleneck(x_concepts)

        c_hat, input_concepts = self._process_concepts(c_hat, c_true, int_idxs)

        ## Selector block ##
        selector_output = self.classifier_selector(latent)
        selector_probs = selector_output['selector_probs'] # [batch_size, memory_size, n_samples]
        selection_dist = selector_output['selection_dist']

        ## Equation execution block ##
        if self.predictor is None:
            raise ValueError("Predictor not initialized. For sr_only mode, run_sr must be called before forward pass.")
        
        predictor_output = self.predictor(selector_probs, input_concepts)

        ## Storing tensors for symbolic regression ##
        if store_tensors:
            self.stored_inputs.append(input_concepts)
            self.stored_outputs.append(predictor_output['y_hat'])
            self.memory_idxs.append(selector_probs)

        return {
            'y_hat': predictor_output['y_hat'],
            'c_hat': c_hat,
            'explanations': predictor_output['explanations'],
            'selection_dist': selection_dist,
            'sampled_memory_idxs': selector_probs
        }

    def loss(self, y_hat, y, c_hat=None, c=None, *args, **kwargs):
        loss = self.concept_based_loss(y_hat, y, c_hat, c)
        return loss
    
    def get_symbolic_equivalent(self, log_dir=None):
        """
        Returns and saves all equations extracted by symbolic regression.
        This includes the complete Pareto front for each target.
        """
        if not hasattr(self, 'all_pareto_equations'):
            raise ValueError("No equations found. Run run_sr first.")
        
        if log_dir is not None:
            import os
            
            os.makedirs(log_dir, exist_ok=True)
            
            # Save all Pareto front equations for each target
            for target_name, equations_list in self.all_pareto_equations.items():
                # Save each equation individually as pickle using store_eq
                for idx, eq_info in enumerate(equations_list):
                    store_eq(eq_info['equation'], log_dir, idx=f"{target_name}_{idx}")