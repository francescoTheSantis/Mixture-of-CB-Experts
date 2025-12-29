import torch.nn as nn
import torch_concepts.nn as pyc_nn
from src.models.baselines.base import BaseModel
from src.models.modules.selector import SelectorModel
from src.models.modules.blackbox_predictor import BlackBoxPredictor
import torch


class MemoryCBM(BaseModel):
    """
    Concept Bottleneck Model with Memory.
    
    This model routes inputs to different black box predictors based on a learned selector.
    Unlike SymbolicRegressorCBM, it does NOT perform symbolic regression or fine-tuning.
    
    Architecture:
    1. Encoder (from BaseModel) extracts latent representations
    2. Concept bottleneck predicts concepts from latent
    3. Selector routes each sample to a memory slot
    4. Black box predictor makes final predictions based on concepts and selection
    """
    
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
        """
        Forward pass through the model.
        
        Args:
            input: Dictionary containing 'x', 'c', 'y'
            
        Returns:
            Dictionary with:
                - y_hat: predictions
                - c_hat: predicted concepts
                - selection_dist: selector distribution over memory slots
                - sampled_memory_idxs: selected memory indices
        """
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
        """Compute the loss for training."""
        loss = self.concept_based_loss(y_hat, y, c_hat, c)

        # L1 regularization on blackbox predictor weights
        l1_norm = sum(p.abs().sum() for p in self.predictor.parameters())
        loss += self.l1_coeff * l1_norm

        return loss
    
    def get_symbolic_equivalent(self, log_dir=None):
        """
        This model doesn't use symbolic regression, so we just return a placeholder.
        This method is kept for compatibility with the evaluation pipeline.
        """
        if log_dir is not None:
            import os
            placeholder_file = os.path.join(log_dir, "no_symbolic_equations.txt")
            with open(placeholder_file, "w") as f:
                f.write("MemoryCBM uses black box predictors only.\n")
                f.write("No symbolic equations are extracted.\n")
