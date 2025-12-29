import torch.nn as nn
import torch_concepts.nn as pyc_nn
from src.models.baselines.base import BaseModel
from src.models.modules.selector import SelectorModel
from src.models.modules.blackbox_predictor import BlackBoxPredictor
import torch


class MemoryCBM(BaseModel):
    """Standard Concept Bottleneck Model with a memory-augmented black box predictor."""
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
                 concept_penalty=1.0,
                 device='cpu',
                 l1_coeff=1e-3,
                 threshold=1e-5,
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

        self.has_concepts = True
        self.y_names = list(y_names)
        self.output_size = output_size
        self.backbone_latent_size = backbone_latent_size
        self.activation = activation
        self.device = device
        self.mc_approx = mc_approx
        self.memory_size = memory_size
        self.l1_coeff = l1_coeff
        self.threshold = threshold
        self.phase = 'standard_training'

        # Instantiate the selector
        self.classifier_selector = SelectorModel(
            input_size=self.backbone_latent_size,
            output_size=self.memory_size,
            n_outputs=self.output_size,
            model_type=selector_model,
            activation=activation,
            decay_rate=decay_rate,
        )
        
        # Concept bottleneck
        self.bottleneck = pyc_nn.LinearConceptBottleneck(
            backbone_latent_size,
            self.c_names,
            activation=nn.Identity(),  # We will later apply a sigmoid if the concept is boolean
        )

        # Black box predictor
        self.predictor = BlackBoxPredictor(
            memory_size=self.memory_size,
            c_names=len(self.c_names),
            output_size=self.output_size,
            activation=activation,
            latent_size=latent_size,
        )

    def forward(self, input):

        latent, x_concepts, c_true, int_idxs = self.encode(input)

        # Concept encoder and concept processing
        c_hat, _ = self.bottleneck(x_concepts)
        c_hat, input_concepts = self._process_concepts(c_hat, c_true, int_idxs)

        # Selector block
        selector_output = self.classifier_selector(latent, global_step=self.global_step)
        selector_probs = selector_output['selector_probs']  # [batch_size, memory_size, n_samples]
        selection_dist = selector_output['selection_dist']

        # Prediction block
        predictor_output = self.predictor(selector_probs, input_concepts)

        return {
            'y_hat': predictor_output['y_hat'],
            'c_hat': c_hat,            
            'selection_dist': selection_dist,
            'sampled_memory_idxs': selector_probs
        }

    def loss(self, y_hat, y, c_hat=None, c=None, *args, **kwargs):
        """Standard CBM loss with L1 regularization on blackbox predictor weights."""
        loss = self.concept_based_loss(y_hat, y, c_hat, c)

        if self.phase == 'standard_training':            
            # L1 regularization on blackbox predictor weights
            l1_norm = sum(p.abs().sum() for p in self.predictor.parameters())
            loss += self.l1_coeff * l1_norm

        return loss

    def cut_weights(self):
        """Set to zero and freeze the weights of the blackbox predictor below a certain threshold."""
        for param in self.predictor.parameters():
            mask = (param.abs() > self.threshold).float()

            # Zero out the masked weights
            with torch.no_grad():
                param.mul_(mask)

            # Freeze only masked elements during backward
            param.register_hook(lambda grad, mask=mask: grad * mask.to(grad.device))

        # From now on we do not want to regularize the weights anymore
        self.phase = 'frozen_weights'
    
    def get_symbolic_equivalent(self, memory_idx=None, return_equations=True):
        """Get the symbolic equivalent of the memory CBM model."""
        predictor_equations = self.predictor.memory_of_predictors[memory_idx].to_symbolic(
            input_names=self.c_names
        )
        return predictor_equations
