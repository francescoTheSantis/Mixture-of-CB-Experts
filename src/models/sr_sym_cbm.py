import torch.nn as nn
import torch_concepts.nn as pyc_nn
from src.models.baselines.base import BaseModel
from src.models.modules.selector import SelectorModel
from src.models.modules.blackbox_predictor import BlackBoxPredictor
from src.models.modules.symbolic_predictor import SymbolicPredictor
from src.utils.expression_utils import store_eq, chain_expression
import numpy as np
import torch
import os

binary_operators = ["*", "+", "-", "/"]
unary_operators = ["sin", "cos", "exp", "log", "tan", "tanh"]
# Added in order to have the same functions of the kan model
extra_functions = {
    "inv": lambda x: 1 / x,
    "square": lambda x: x**2,
    "cube": lambda x: x**3,
    #"x^4": lambda x: x**4,
    #"x^5": lambda x: x**5,
    "inv2": lambda x: 1 / x**2,
    "inv3": lambda x: 1 / x**3,
    #"inv4": lambda x: 1 / x**4,
    #"inv5": lambda x: 1 / x**5,
    "sqrt": lambda x: x**0.5,
    #"x^1.5": lambda x: x**1.5,
    "invsqrt": lambda x: 1 / x**0.5,
    #"abs": lambda x: abs(x),
    #"sgn": lambda x: 1 if x > 0 else (-1 if x < 0 else 0),
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
            # 'timeout_in_seconds': 60 * 3,
            'maxsize': 40,  # Limit the size of the equations.
            'maxdepth': 40,  # Limit the depth of the equations to maxsize so that the most complex expression tree is a chain (easy to compute).
        }

        # Instantiate the predictor
        self.predictor = BlackBoxPredictor(
            memory_size=self.memory_size,
            c_names=len(self.c_names),
            output_size=self.output_size,
            activation=activation,
            latent_size=latent_size,
        )

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

        return {
            'y_hat': predictor_output['y_hat'],
            'c_hat': c_hat,
            'explanations': predictor_output['explanations'],
            'selection_dist': selection_dist,
            'sampled_memory_idxs': selector_probs
        }
    
    def symbolic_substitution(self, equations):
        """
        Substitute symbolic equations into the model's predictor.
        
        Args:
            equations (dict): A dictionary where keys are memory slot indices and 
                              values are dictionaries mapping output names to 
                              sympy equations.
        """
        self.predictor = SymbolicPredictor(
            equations=equations,
            c_names=self.c_names,
        )

        for p in self.predictor.parameters():
            p.requires_grad = False  # Freeze all parameters
        
        self.predictor = self.predictor.to(self.device)

    def loss(self, y_hat, y, c_hat=None, c=None, *args, **kwargs):
        loss = self.concept_based_loss(y_hat, y, c_hat, c)
        return loss
    
    def get_symbolic_equivalent(self, log_dir=None):
        """
        Returns and saves all equations extracted by symbolic regression.
        This includes the complete Pareto front for each target.
        """

        # According to our metric the most complex equation is represented by an expression tree
        # which is a chain of operations of size equal to maxsize.
        equation = chain_expression(self.pysr_params['maxsize'])
        if log_dir is not None:
            store_eq(equation, log_dir)

        # Get equations for each memory slot
        if log_dir is not None:
            memory_eq_dir = os.path.join(log_dir, "memory_slots")
            os.makedirs(memory_eq_dir, exist_ok=True)
            self._store_memory_equations(memory_eq_dir)

    def _store_memory_equations(self, dir):
        """
        Store the equations associated to each memory slot.
        """
        # Check if predictor has equations (i.e., if it's a SymbolicPredictor)
        if hasattr(self.predictor, 'trainable_equations'):
            # Store equations in both .pkl and text format
            for mem_idx, set_name in enumerate(sorted(self.predictor.trainable_equations.keys())):
                mem_dir = os.path.join(dir, f"memory_slot_{mem_idx}")
                os.makedirs(mem_dir, exist_ok=True)
                
                # Create text file for this memory slot
                text_file = os.path.join(mem_dir, "equations.txt")
                with open(text_file, "w") as f:
                    f.write(f"Memory Slot {mem_idx} (Set: {set_name})\n")
                    f.write("=" * 60 + "\n\n")
                    
                    # Store each equation in this memory slot
                    for eq_idx, eq_name in enumerate(self.predictor.equation_names[set_name]):
                        eq_module = self.predictor.trainable_equations[set_name][eq_name]
                        
                        # Get the equation expression
                        equation_expr = eq_module.sympy_expr
                        
                        # Store in pickle format
                        store_eq(equation_expr, mem_dir, idx=eq_idx)
                        
                        # Write to text file
                        f.write(f"Equation {eq_idx} ({eq_name}):\n")
                        f.write(f"  Expression: {equation_expr}\n")
                        f.write(f"  Parameters: {eq_module.get_param_values()}\n")
                        f.write(f"  Current form: {eq_module.get_equation_string()}\n")
                        f.write("\n")
        else:
            # Predictor doesn't have symbolic equations yet
            no_equations_file = os.path.join(dir, "no_equations.txt")
            with open(no_equations_file, "w") as f:
                f.write("No symbolic equations available in memory yet.\n")
                f.write("The predictor may be a BlackBoxPredictor or not yet trained.\n")