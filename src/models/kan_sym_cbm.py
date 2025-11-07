import torch.nn as nn
import torch_concepts.nn as pyc_nn
from src.models.baselines.base import BaseModel
from src.utils.expression_utils import kan_expression, store_eq
from src.models.modules.selector import SelectorModel
from src.models.modules.kan_predictor import KANPredictor

class KANSymbolicCBM(BaseModel):
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
                 encoder=None,
                 mc_approx=1,
                 selector_model='linear',
                 concept_loss_form=nn.BCELoss(),
                 backbone_latent_size=None,
                 concept_type='binary',
                 known_equations=None,
                 disjoint_training=False,
                 decay_rate='cosine',
                 embedding_memory=False,
                 concept_penalty=1.0,
                 regularize=False,
                 widths=None,
                 device='cpu',
                 speed_up_training=False,
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
        self.regularize = regularize
        self.symbolic_predictors = False
        self.device = device
        self.speed_up_training = speed_up_training

        if widths is not None:
            self.widths = widths
        else:
            if self.output_size == 1:
                # Approach suggested by the authors of KAN
                self.widths = [len(self.c_names), len(self.c_names)+1, self.output_size]
            else:
                # For multi-output tasks, we use a smaller architecture since the number of parameters 
                # grows quickly with the number of outputs and this slows down the auto-symbolic search.
                # NOTE: this is a design choice made according to the hardware and different architectures can be used.
                self.widths = [len(self.c_names), self.output_size]

        grid_size = 5
        k = 3

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

        # KAN predictor instantiation
        self.kan_layers = KANPredictor(
            widths=self.widths,
            grid=grid_size,
            k=k,
            memory_size=self.memory_size,
            device=self.device,
            speed_up_training=self.speed_up_training
        )

    def setup_kan_grid(self, grid_inputs):
        self.kan_layers.setup_kan_grid(grid_inputs)

    def allow_symbolic(self):
        self.kan_layers.allow_symbolic()
    
    def get_learned_equations(self, log_dir):
        self.kan_layers.get_learned_equations(log_dir)

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
        kan_predictor_output = self.kan_layers(selector_probs, input_concepts)

        return {
            'y_hat': kan_predictor_output['y_hat'],
            'c_hat': c_hat,
            'explanations': kan_predictor_output['explanations'],
            'selection_dist': selection_dist,
            'sampled_memory_idxs': selector_probs
        }

    def loss(self, y_hat, y, c_hat=None, c=None, *args, **kwargs):
        loss = self.concept_based_loss(y_hat, y, c_hat, c)
        # KAN regularization: it promotes sparsity in the KAN layers
        loss += self.kan_layers.regularization_term()
        return loss
    
    def get_symbolic_equivalent(self, log_dir=None):
        """
        Returns the equation associated to the KAN predictor of the model
        """

        # Remove the last element in widths and substitute with 1
        # This is because we want to get the symbolic expression for a single output (as we did for the other models).
        try:
            single_output_widths = self.widths[:-1] + [1]
            equation = kan_expression(single_output_widths)
        except ValueError:
            print(f"Zeros are appended to the each element in widths, we need to remove them")
            self.widths = [w[0] for w in self.widths]
            single_output_widths = self.widths[:-1] + [1]
            equation = kan_expression(single_output_widths)

        # Generate the abstract (operators are not defined) symbolic equivalent of the kan used by the model.
        store_eq(equation, log_dir)
