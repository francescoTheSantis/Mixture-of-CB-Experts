import torch.nn as nn
import torch_concepts.nn as pyc_nn
from src.models.baselines.base import BaseModel
from src.models.modules.selector import SelectorModel
from src.models.modules.blackbox_predictor import BlackBoxPredictor
from src.models.modules.symbolic_predictor import SymbolicPredictor
from src.utils.expression_utils import store_eq, chain_expression
import sympy
import torch
import os


# Sympy Function subclasses so expressions display as AND/OR/NOT
class AND(sympy.Function):
    @classmethod
    def eval(cls, x, y):
        return None  # prevent eager evaluation — keep symbolic

class OR(sympy.Function):
    @classmethod
    def eval(cls, x, y):
        return None

class NOT(sympy.Function):
    @classmethod
    def eval(cls, x):
        return None


# Boolean operators defined for {0,1} inputs
boolean_binary_operators = [
    "AND(x, y) = x * y",
    "OR(x, y) = x + y - x * y",
]

boolean_unary_operators = [
    "NOT(x) = one(x) - x",
]

boolean_extra_sympy_mappings = {
    "AND": AND,
    "OR": OR,
    "NOT": NOT,
}


class BooleanSymbolicCBM(BaseModel):
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
                 pysr_params=None,
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
            n_outputs=self.output_size,
            model_type=selector_model,
            activation=activation,
            decay_rate=decay_rate,
        )
        
        self.bottleneck = pyc_nn.LinearConceptBottleneck(
            backbone_latent_size,
            self.c_names,
            activation=nn.Identity(),
        )

        # PySR parameters for Boolean expression discovery
        size = len(self.c_names) * 5
        self.pysr_params = {
            'optimizer_iterations': 8, 
            'populations': 40,
            'population_size': 60,
            'niterations': 100,
            'ncycles_per_iteration': 380,
            'maxsize': size,
            'maxdepth': size,
            # Boolean operators only
            'binary_operators': boolean_binary_operators,
            'unary_operators': boolean_unary_operators,
            'extra_sympy_mappings': boolean_extra_sympy_mappings,
            # Custom loss: margin-like for {-1, 1} targets
            'elementwise_loss': "myloss(prediction, target) = (1 - prediction * target)^2",
            # No constants
            'should_optimize_constants': False,
            'complexity_of_constants': 100,
            'timeout_in_seconds': 60,
            'early_stop_condition': 1e-5
        }

        # Override with user-defined params
        if pysr_params is not None:
            pysr_params = {k: v for k, v in pysr_params.items() if k != 'name'}
            self.pysr_params.update(pysr_params)

        # Instantiate the predictor: always MLP (not linear), even for classification
        self.predictor = BlackBoxPredictor(
            memory_size=self.memory_size,
            c_names=len(self.c_names),
            output_size=self.output_size,
            activation=activation,
            latent_size=latent_size,
            linear=False,
        )

    ###### Forward and loss methods ######
    def forward(self, input, store_for_finetuning=False):

        latent, x_concepts, c_true, int_idxs = self.encode(input)

        ## Concept encoder and concept processing block ##
        c_hat, _ = self.bottleneck(x_concepts)

        c_hat, input_concepts = self._process_concepts(c_hat, c_true, int_idxs)

        ## Selector block ##
        selector_output = self.classifier_selector(latent, global_step=self.global_step)
        selector_probs = selector_output['selector_probs']
        selection_dist = selector_output['selection_dist']

        ## Equation execution block ##
        if self.predictor is None:
            raise ValueError("Predictor not initialized.")

        predictor_output = self.predictor(selector_probs, input_concepts)

        return {
            'y_hat': predictor_output['y_hat'],
            'c_hat': c_hat,        
            'input_concepts': input_concepts,   
            'selection_dist': selection_dist,
            'sampled_memory_idxs': selector_probs
        }

    def symbolic_substitution(self, equations):
        """
        Replace the BlackBoxPredictor with a SymbolicPredictor
        containing the discovered Boolean expressions.
        No affine parameters — Boolean rules are used as-is.
        """
        self.predictor = SymbolicPredictor(
            equations=equations,
            c_names=self.c_names,
        )

        for p in self.predictor.parameters():
            p.requires_grad = False
        
        self.predictor = self.predictor.to(self.device)

    def loss(self, y_hat, y, c_hat=None, c=None, *args, **kwargs):
        loss = self.concept_based_loss(y_hat, y, c_hat, c)
        return loss
    
    def get_symbolic_equivalent(self, log_dir=None):
        """
        Returns and saves all equations extracted by symbolic regression.
        """
        equation = chain_expression(self.pysr_params['maxsize'])
        if log_dir is not None:
            store_eq(equation, log_dir)

        if log_dir is not None:
            memory_eq_dir = os.path.join(log_dir, "memory_slots")
            os.makedirs(memory_eq_dir, exist_ok=True)
            self._store_memory_equations(memory_eq_dir)

    def _store_memory_equations(self, dir):
        """Store the equations associated to each memory slot."""
        if hasattr(self.predictor, 'trainable_equations'):
            for mem_idx, set_name in enumerate(sorted(self.predictor.trainable_equations.keys())):
                mem_dir = os.path.join(dir, f"memory_slot_{mem_idx}")
                os.makedirs(mem_dir, exist_ok=True)
                
                text_file = os.path.join(mem_dir, "equations.txt")
                with open(text_file, "w") as f:
                    f.write(f"Memory Slot {mem_idx} (Set: {set_name})\n")
                    f.write("=" * 60 + "\n\n")
                    
                    for eq_idx, eq_name in enumerate(self.predictor.equation_names[set_name]):
                        eq_module = self.predictor.trainable_equations[set_name][eq_name]
                        equation_expr = eq_module.sympy_expr
                        store_eq(equation_expr, mem_dir, idx=eq_idx)
                        
                        f.write(f"Equation {eq_idx} ({eq_name}):\n")
                        f.write(f"  Expression: {equation_expr}\n")
                        f.write(f"  Parameters: {eq_module.get_param_values()}\n")
                        f.write(f"  Current form: {eq_module.get_equation_string()}\n")
                        f.write("\n")
        else:
            no_equations_file = os.path.join(dir, "no_equations.txt")
            with open(no_equations_file, "w") as f:
                f.write("No symbolic equations available in memory yet.\n")
                f.write("The predictor may be a BlackBoxPredictor or not yet trained.\n")
