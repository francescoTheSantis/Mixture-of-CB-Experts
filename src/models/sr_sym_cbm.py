import torch.nn as nn
import torch_concepts.nn as pyc_nn
from src.models.baselines.base import BaseModel
from src.models.modules.selector import SelectorModel
from src.models.modules.blackbox_predictor import BlackBoxPredictor
from src.models.modules.symbolic_predictor import SymbolicPredictor
import sympy as sp
from src.utils.expression_utils import store_eq
import pysr # Ensure pysr is imported
from pysr import PySRRegressor
import numpy as np
import torch
import os
os.environ["JULIA_NUM_THREADS"] = "1"

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
            'extra_sympy_mappings': extra_functions,
            'elementwise_loss': "loss(prediction, target) = (prediction - target)^2",
            'complexity_of_constants': 1,
            'timeout_in_seconds': 60 * 2, # Stop after 2 minutes
        }

        # Instantiate the predictor
        self.predictor = BlackBoxPredictor(
            memory_size=self.memory_size,
            c_names=len(self.c_names),
            output_size=self.output_size,
            activation=activation,
            latent_size=latent_size,
        )
        
        # Storage for fine-tuning
        self.stored_concepts = []
        self.stored_targets = []
        self.stored_selector_probs = []

    def reset_stored_tensors(self):
        self.stored_concepts = []
        self.stored_targets = []
        self.stored_selector_probs = []

    ###### Forward and loss methods ######
    def forward(self, input, store_for_finetuning=False):

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
            raise ValueError("Predictor not initialized.")
        
        predictor_output = self.predictor(selector_probs, input_concepts)
        
        ## Store data for fine-tuning if requested ##
        if store_for_finetuning:
            # Store concepts (detached from computation graph)
            self.stored_concepts.append(input_concepts.detach().cpu())
            # Store targets
            self.stored_targets.append(predictor_output['y_hat'].detach().cpu())
            # Store selector probabilities (which memory slot was selected)
            # We take argmax to get the selected memory slot for each sample
            selected_memory = selector_probs.argmax(dim=1).detach().cpu()  # [batch_size, n_samples]
            self.stored_selector_probs.append(selected_memory)

        return {
            'y_hat': predictor_output['y_hat'],
            'c_hat': c_hat,
            'explanations': predictor_output['explanations'],
            'selection_dist': selection_dist,
            'sampled_memory_idxs': selector_probs
        }
    
    def run_symbolic_finetuning(self):
        """
        Fine-tune the model by replacing each MLP in the BlackBoxPredictor with 
        symbolic equations discovered by PySR.
        
        This method:
        1. Collects stored concepts, targets, and selector probabilities
        2. For each memory slot and each output class, fits a PySR model
        3. Extracts the best equation
        4. Creates a SymbolicPredictor with all discovered equations
        """
        
        if len(self.stored_concepts) == 0:
            raise ValueError("No stored data for fine-tuning. Run forward passes with store_for_finetuning=True first.")
        
        # Concatenate all stored data
        all_concepts = torch.cat(self.stored_concepts, dim=0).numpy()  # [N, n_concepts]
        all_targets = torch.cat(self.stored_targets, dim=0).numpy()    # [N, ] or [N, n_outputs]
        all_selector_probs = torch.cat(self.stored_selector_probs, dim=0)  # [N, n_samples]
        
        # Handle different target shapes
        if all_targets.ndim == 1:
            all_targets = all_targets.reshape(-1, 1)
        
        # Dictionary to store equations
        # Structure: {memory_idx: {output_name: sympy_equation}}
        all_equations = {i: {} for i in range(self.memory_size)}
        
        # For each memory slot
        for memory_idx in range(self.memory_size):
            # Get samples where this memory slot was selected
            # We take the most common selection across n_samples
            memory_mask = (all_selector_probs.mode(dim=1).values == memory_idx).numpy()
            n_samples_for_memory = memory_mask.sum()
            
            # Filter concepts and targets for this memory slot
            X_memory = all_concepts[memory_mask]  # [n_samples_memory, n_concepts]
            y_memory = all_targets[memory_mask]    # [n_samples_memory, n_outputs]
            
            # Skip if no samples for this memory slot
            if n_samples_for_memory == 0:
                print(f"Memory slot {memory_idx} has no samples. Skipping.")
                for output_idx in range(self.output_size):
                    output_name = self.y_names[output_idx] if output_idx < len(self.y_names) else f"y_{output_idx}"
                    all_equations[memory_idx][output_name] = sp.sympify("0")
                continue
            
            # Note: you are running with more than 10,000 datapoints. 
            # You should consider turning on batching (`options.batching`), and also if you need that many datapoints. 
            # Unless you have a large amount of noise (in which case you should smooth your dataset first), 
            # generally < 10,000 datapoints is enough to find a functional form.
            # Given the message returned by PySR, we can subsample if needed.
            subsample_size = 1000
            if n_samples_for_memory > subsample_size:
                print(f"Subsampling to {subsample_size} for PySR.")
                indices = np.random.choice(n_samples_for_memory, size=subsample_size, replace=False)
                X_memory = X_memory[indices]
                y_memory = y_memory[indices]

            # For each output
            for output_idx in range(self.output_size):
                output_name = self.y_names[output_idx] if output_idx < len(self.y_names) else f"y_{output_idx}"
                                
                y_target = y_memory[:, output_idx]
                
                # Validate data: check for NaN and Inf values
                if np.any(np.isnan(X_memory)) or np.any(np.isinf(X_memory)):
                    print(f"WARNING: X_memory contains NaN or Inf values for memory {memory_idx}. Cleaning data.")
                    valid_mask = ~(np.isnan(X_memory).any(axis=1) | np.isinf(X_memory).any(axis=1))
                    X_memory = X_memory[valid_mask]
                    y_target = y_target[valid_mask]
                
                if np.any(np.isnan(y_target)) or np.any(np.isinf(y_target)):
                    print(f"WARNING: y_target contains NaN or Inf values for memory {memory_idx}, output {output_name}. Cleaning data.")
                    valid_mask = ~(np.isnan(y_target) | np.isinf(y_target))
                    X_memory_clean = X_memory[valid_mask]
                    y_target = y_target[valid_mask]
                else:
                    X_memory_clean = X_memory
                
                # Check if we have enough samples after cleaning
                if len(y_target) < 10:
                    print(f"WARNING: Too few valid samples ({len(y_target)}) for memory {memory_idx}, output {output_name}. Using mean as fallback.")
                    if len(y_target) == 0:
                        all_equations[memory_idx][output_name] = sp.sympify("0")
                    else:
                        all_equations[memory_idx][output_name] = sp.sympify(str(y_target.mean()))
                    continue
                
                print(f"\nFitting PySR for output '{output_name}' (memory slot {memory_idx})...")
                print(f"  Input shape: {X_memory_clean.shape}, Target shape: {y_target.shape}")
                
                try:
                    model = PySRRegressor(
                        **self.pysr_params,
                        verbosity=0,  # Reduce verbosity to avoid Julia output issues
                        progress=True,
                    )
                    
                    model.fit(X_memory_clean, y_target)
                    
                    # Get the best equation (highest score)
                    equations_df = model.equations_
                    print(f"  Pareto front has {len(equations_df)} equations")
                    
                    # Select the best equation by score
                    best_eq_row = equations_df.nlargest(1, 'score').iloc[0]
                    sympy_eq = best_eq_row['sympy_format']
                    
                    # Rename variables from x0, x1, ... to concept names
                    for i, c_name in enumerate(self.c_names):
                        sympy_eq = sympy_eq.subs(sp.Symbol(f'x{i}'), sp.Symbol(c_name))
                    
                    all_equations[memory_idx][output_name] = sympy_eq
                    
                    print(f"  ✓ Best equation: {sympy_eq}")
                    print(f"    Loss: {best_eq_row['loss']:.6f}")
                    print(f"    Complexity: {best_eq_row['complexity']}")
                    print(f"    Score: {best_eq_row['score']:.6f}")

                    # Clean up the model
                    del model
                    
                except Exception as e:
                    print(f"  ✗ ERROR fitting PySR for memory {memory_idx}, output {output_name}:")
                    print(f"    {type(e).__name__}: {str(e)}")
                    print(f"    Using fallback constant equation (mean value)")
                    if len(y_target) == 0:
                        print(f"    No samples available. Using zero as fallback.")
                        all_equations[memory_idx][output_name] = sp.sympify("0")
                    else:
                        mean_val = float(y_target.mean())
                        all_equations[memory_idx][output_name] = sp.sympify(str(mean_val))
                        print(f"    Fallback equation: {mean_val}")

        # Verify all equations are present
        for memory_idx in range(self.memory_size):
            for output_idx in range(self.output_size):
                output_name = self.y_names[output_idx] if output_idx < len(self.y_names) else f"y_{output_idx}"
                if output_name not in all_equations[memory_idx]:
                    raise ValueError(f"Missing equation for memory {memory_idx}, output {output_name}")
        
        # Replace the BlackBoxPredictor with SymbolicPredictor
        self.predictor = SymbolicPredictor(
            equations=all_equations,
            c_names=self.c_names,
        )
        
        # Move predictor to the correct device
        self.predictor = self.predictor.to(self.device)
        
        # Clear stored data to free memory
        self.reset_stored_tensors()

    def loss(self, y_hat, y, c_hat=None, c=None, *args, **kwargs):
        loss = self.concept_based_loss(y_hat, y, c_hat, c)
        return loss
    
    def get_symbolic_equivalent(self, log_dir=None):
        """
        Returns and saves all equations extracted by symbolic regression.
        This includes the complete Pareto front for each target.
        """
        pass