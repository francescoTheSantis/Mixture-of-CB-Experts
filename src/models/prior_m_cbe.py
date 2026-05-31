import torch.nn as nn
import torch_concepts.nn as pyc_nn
from src.models.baselines.base import BaseModel
from src.models.modules.selector import SelectorModel
from src.models.modules.symbolic_predictor import SymbolicPredictor
from src.utils.expression_utils import store_eq
import sympy as sp
import os

class PriorMCBE(BaseModel):
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
                 hard_concepts=False,
                 encoder=None,
                 mc_approx=1,
                 selector_model='linear',
                 backbone_latent_size=None,
                 concept_type='binary',
                 known_equations=None,
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

        # For this set of experiments, we assume we have access to known equations only for regression tasks.
        # Convert from list format to SymbolicPredictor format
        # Input format: [eq1, eq2, ...] - list of equations (each is a memory slot)
        # SymbolicPredictor format: {'set0': {'eq0': task_eq1, ...}, 'set1': {'eq0': task_eq1, ...}, ...}
        # Where each 'set' is a memory slot, and each memory slot contains equations for all tasks
        
        # For regression with single output, we typically have one equation per memory slot
        # But the structure needs to support multiple task outputs
        # Assuming known_equations is a list where each element is an equation for a different memory slot
        # and we have a single task output
        
        symbolic_equations = {}
        for memory_idx, equation in enumerate(known_equations):
            set_name = f'set{memory_idx}'
            symbolic_equations[set_name] = {}
            # Single task output per memory slot
            eq_name = self.y_names[0] 
            # Convert string equation to sympy expression if needed
            if isinstance(equation, str):
                # Create local dict with concept names as symbols
                local_dict = {f'c{i}': sp.Symbol(name) for i, name in enumerate(c_names)}
                symbolic_equations[set_name][eq_name] = sp.sympify(equation, locals=local_dict)
            else:
                symbolic_equations[set_name][eq_name] = equation
        
        # memory_size is the number of memory slots (number of equations)
        self.memory_size = len(known_equations)

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
            activation=nn.Identity(), # we will later apply a sigmoid if the concept is boolean
        )

        # SymbolicPredictor instantiation with frozen parameters
        self.prior_predictor = SymbolicPredictor(
            equations=symbolic_equations,
            c_names=self.c_names
        )
        
        # Freeze all parameters of the SymbolicPredictor
        for param in self.prior_predictor.parameters():
            param.requires_grad = False

    ###### Forward and loss methods ######
    def forward(self, input):

        latent, x_concepts, c_true, int_idxs = self.encode(input)

        ## Concept encoder and concept processing block ##
        c_hat, _ = self.bottleneck(x_concepts)

        c_hat, input_concepts = self._process_concepts(c_hat, c_true, int_idxs)

        ## Selector block ##
        selector_output = self.classifier_selector(latent, global_step=self.global_step)
        selector_probs = selector_output['selector_probs'] # [batch_size, memory_size, n_samples]
        selection_dist = selector_output['selection_dist']

        ## Equation execution block ##
        prior_predictor_output = self.prior_predictor(selector_probs, input_concepts)

        return {
            'y_hat': prior_predictor_output['y_hat'],
            'c_hat': c_hat,
            'selection_dist': selection_dist,
            'sampled_memory_idxs': selector_probs
        }

    def loss(self, y_hat, y, c_hat=None, c=None, *args, **kwargs):
        loss = self.concept_based_loss(y_hat, y, c_hat, c)
        return loss
    
    def get_symbolic_equivalent(self, log_dir=None):
        """
        Returns the most complex equation stored in the memory.
        """
        # Find the most complex equation across all memory slots
        max_complexity = 0
        most_complex_eq = None
        
        for set_name in self.prior_predictor.trainable_equations.keys():
            for eq_name in self.prior_predictor.equation_names[set_name]:
                eq_module = self.prior_predictor.trainable_equations[set_name][eq_name]
                eq_expr = eq_module.sympy_expr
                
                # Calculate complexity as the number of nodes in the expression tree
                complexity = sum(1 for _ in sp.preorder_traversal(eq_expr))
                
                if complexity > max_complexity:
                    max_complexity = complexity
                    most_complex_eq = eq_expr
        
        # Store the most complex equation
        if most_complex_eq is not None and log_dir is not None:
            store_eq(most_complex_eq, log_dir)
        
        # Store equations for each memory slot
        if log_dir is not None:
            memory_eq_dir = os.path.join(log_dir, "memory_slots")
            os.makedirs(memory_eq_dir, exist_ok=True)
            self._store_memory_equations(memory_eq_dir)
    
    def _store_memory_equations(self, dir):
        """
        Store the prior equations associated to each memory slot.
        """
        # Store equations for each memory slot
        for mem_idx, set_name in enumerate(sorted(self.prior_predictor.trainable_equations.keys())):
            mem_dir = os.path.join(dir, f"memory_slot_{mem_idx}")
            os.makedirs(mem_dir, exist_ok=True)
            
            # Create text file for this memory slot
            text_file = os.path.join(mem_dir, "equations.txt")
            with open(text_file, "w") as f:
                f.write(f"Memory Slot {mem_idx} (Set: {set_name})\n")
                f.write("=" * 60 + "\n\n")
                
                # Store each equation in this memory slot
                for eq_idx, eq_name in enumerate(self.prior_predictor.equation_names[set_name]):
                    eq_module = self.prior_predictor.trainable_equations[set_name][eq_name]
                    
                    # Get the equation expression
                    equation_expr = eq_module.sympy_expr
                    
                    # Store in pickle format
                    store_eq(equation_expr, mem_dir, idx=eq_idx)
                    
                    # Write to text file
                    f.write(f"Equation {eq_idx} ({eq_name}):\n")
                    f.write(f"  Original Expression: {equation_expr}\n")
                    f.write(f"  Parameters: {eq_module.get_param_values()}\n")
                    f.write(f"  Current form: {eq_module.get_equation_string()}\n")
                    
                    # Calculate and display complexity
                    complexity = sum(1 for _ in sp.preorder_traversal(equation_expr))
                    f.write(f"  Complexity (node count): {complexity}\n")
                    f.write("\n")