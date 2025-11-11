import torch.nn as nn
import torch_concepts.nn as pyc_nn
from src.models.baselines.base import BaseModel
from src.models.modules.selector import SelectorModel
from src.models.modules.symbolic_predictor import SymbolicPredictor
from src.models.modules.sr_predictor import SRPredictor
import sympy as sp
from src.utils.expression_utils import store_eq

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

        # SymbolicPredictor instantiation with frozen parameters
        self.predictor = SRPredictor(
            memory_size=self.memory_size,
            c_names=self.c_names,
            pysr_params={
                'niterations': 1000,
                'binary_operators': ['plus', 'sub', 'mul', 'div'],
                'unary_operators': ['sin', 'cos', 'exp', 'log', 'neg'],
                'popsize': 1000,
                'maxsize': 30,
                'maxdepth': 5,
                'loss': 'L2Dist',
                'complexity_of_constants': 0.1,
            },
        )

    def reset_stored_tensors(self):
        self.stored_inputs = []
        self.stored_outputs = []
        self.memory_idxs = []

    ###### Forward and loss methods ######
    def forward(self, input):

        latent, x_concepts, c_true, int_idxs = self.encode(input)

        ## Concept encoder and concept processing block ##
        c_hat, _ = self.bottleneck(x_concepts)

        c_hat, input_concepts = self._process_concepts(c_hat, c_true, int_idxs)

        ## Selector block ##
        selector_output = self.classifier_selector(latent)
        selector_probs = selector_output['selector_probs'] # [batch_size, memory_size, n_samples]
        selection_dist = selector_output['selection_dist']

        ## Equation execution block ##
        predictor_output = self.predictor(selector_probs, input_concepts)

        ## Storing tensors for symbolic regression ##
        if self.store_tensors:
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
        self.predictor.get_symbolic_equivalent(log_dir)