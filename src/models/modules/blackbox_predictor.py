from torch import nn
from src.models.encoders.mlp import MLPEncoder
from src.utils.expression_utils import store_eq
import torch

class BlackBoxPredictor(nn.Module):
    def __init__(self, 
                 memory_size, 
                 c_names,
                 output_size,
                 activation,
                 latent_size,
                ):
        super(BlackBoxPredictor, self).__init__()

        self.memory_size = memory_size
        self.c_names = c_names
        self.output_size = output_size
        self.show_explanations = False
        self.activation = activation
        self.latent_size = latent_size

        # Create a module of mlp encoders for each memory slot
        memory_of_predictors = nn.ModuleList()
        for _ in range(memory_size):
            mlp = MLPEncoder(
                input_size=c_names,
                output_size=output_size,
                hidden_size=latent_size,
                activation=activation,
                num_layers=1, # one hidden layer
            )
            memory_of_predictors.append(mlp)
        self.memory_of_predictors = memory_of_predictors

    def _get_explanations(self, prob_per_classifier, y_hat):
        # TODO: to be completed once the kan implementation is stable
        if self.show_explanations:
            if not self.equations_for_explanations_ready:
                self._setup_string_equations()
            explanations = None
        else:
            explanations = [None] * prob_per_classifier.size(0)

        return explanations

    def forward(self, prob_per_classifier, input_concepts):
        bsz = input_concepts.shape[0]
        eq_outputs = []
        for _, mlp in enumerate(self.memory_of_predictors):
            eq_outputs.append(
                mlp(input_concepts)
            )
            
        # Stack the outputs along the class dimension
        eq_outputs = torch.stack(eq_outputs, dim=1) # shape: (bsz, memory_size, n_targets)

        y_hat = torch.einsum('bms,bmt->bts', prob_per_classifier, eq_outputs)

        # Get the explanations (the selected equations)
        if self.training:
            explanations = [None] * bsz
        else:
            explanations = self._get_explanations(prob_per_classifier, y_hat)

        return {
            'y_hat': y_hat,
            'explanations': explanations,
        }