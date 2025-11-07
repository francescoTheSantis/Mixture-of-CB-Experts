import torch.nn as nn
import torch_concepts.nn as pyc_nn
from src.models.baselines.base import BaseModel
from src.utils.expression_utils import linear_classifier_expression, store_eq
from src.models.modules.selector import SelectorModel
from src.models.modules.linear_predictor import LinearPredictor

class LinearSymbolicCBM(BaseModel):
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
                 embedding_size = 16,
                 latent_size = 128,
                 c_groups=None,
                 memory_size=7,
                 hard_concepts=False,
                 weight_reg=0,
                 encoder=None,
                 mc_approx=1,
                 embedding_memory=True,
                 selector_model='linear',
                 concept_loss_form=nn.BCELoss(),
                 multiple_eq_per_classifier=True,
                 backbone_latent_size=None,
                 concept_type='binary',
                 decay_rate='cosine',
                 bias=None,
                 disjoint_training=False,
                 concept_penalty=1.0,
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
        self.multiple_eq_per_classifier = multiple_eq_per_classifier

        self.mc_approx = mc_approx
        self.embedding_memory = embedding_memory
        self.memory_size = memory_size
        self.selector_model = selector_model
        self.decay_rate = decay_rate

        self.bias = None if bias is None else bias

        if self.bias not in [None, 'local', 'global']:
            raise ValueError("Invalid bias type. Expected one of [None, 'local', 'global'].")

        self.bottleneck = pyc_nn.LinearConceptBottleneck(
            backbone_latent_size,
            self.c_names,
            activation=nn.Identity(), # we will later apply a sigmoid if the concept is boolean
        )

        # The selector generates logits that define a probability distribution 
        # over the linear equations stored in memory.
        selector_input_size = backbone_latent_size
        selector_output_size = memory_size
        
        # Instantiate the selector
        self.classifier_selector = SelectorModel(
            input_size=selector_input_size,
            output_size=selector_output_size,
            model_type=self.selector_model,
            activation=activation,
            decay_rate=self.decay_rate,
        )

        # Instantiate the linear memory predictor
        self.linear_memory_predictor = LinearPredictor(
            memory_size=memory_size,
            latent_size=latent_size,
            c_names=c_names,
            bias=self.bias,
            y_names=y_names,
            activation=activation,
        )

    def forward(self, input):
        latent, latent_concepts, c_true, int_idxs = self.encode(input)
        bsz = latent.shape[0]

        c_hat, _ = self.bottleneck(latent_concepts)

        c_hat, input_concepts = self._process_concepts(c_hat, c_true, int_idxs)

        # Get the distribution over the memory of linear equations
        selector_output = self.classifier_selector(
            latent,
            mc_approx=self.mc_approx,
            global_step=self.global_step
        )
        selector_probs = selector_output['selector_probs']
        selection_dist = selector_output['selection_dist']

        linear_output = self.linear_memory_predictor(
            selector_probs,
            input_concepts
        )
        y_hat = linear_output['y_hat']
        predicted_weights = linear_output['explanations']

        return {
            'y_hat': y_hat,
            'c_hat': c_hat,
            'explanations': predicted_weights,
            'selection_dist': selection_dist
        }
        
    def loss(self, y_hat, y, c_hat=None, c=None, *args, **kwargs):
        loss = self.concept_based_loss(y_hat, y, c_hat, c)

        # Collect all the parameters in the memory
        params = self.linear_memory_predictor.equation_decoder(
            self.linear_memory_predictor.equation_memory.weight
        )

        if self.bias == 'local':
            weights = params[:,:,:-1]
            bias = params[:,:,-1]
        elif self.bias == None:
            weights = params
            bias = None
        elif self.bias == 'global':
            weights = params
            bias = self.linear_memory_predictor.bias_params

        # L1 Regularization over weights
        loss += self.weight_reg * weights.abs().sum()

        # L2 regularization over the bias
        if bias != None:
            loss += self.weight_reg * bias.pow(2).sum()

        return loss

    def get_symbolic_equivalent(self, log_dir=None):
        """
        Returns the equation associated to the predictor of the model
        """
        
        # Return the most complex linear equation that can obtained after training (all concepts are relevant) 
        bias = True if self.bias != None else False
        equation = linear_classifier_expression(len(self.c_names), include_bias=bias)
        store_eq(equation, log_dir)