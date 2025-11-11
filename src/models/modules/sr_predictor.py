from torch import nn
from src.models.encoders.mlp import MLPEncoder
from src.utils.expression_utils import store_eq
import torch

class SRPredictor(nn.Module):
    def __init__(self, 
                 memory_size, 
                 c_names,
                 output_size,
                 pysr_params,
                 activation,
                ):
        super(SRPredictor, self).__init__()

        self.memory_size = memory_size
        self.c_names = c_names
        self.output_size = output_size
        self.pysr_params = pysr_params

        # Create a module of mlp encoders for each memory slot
        memory_of_predictors = nn.ModuleList()
        for _ in range(memory_size):
            mlp = MLPEncoder(
                input_size=c_names,
                output_size=output_size,
                hidden_size=c_names,
                activation=activation,
                num_layers=2,
            )
            memory_of_predictors.append(mlp)
        self.memory_of_predictors = memory_of_predictors

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

    def get_symbolic_equivalent(self, log_dir=None):
        """
        Returns the equation associated to the predictor of the model
        """

        # Get as many equations as the output size
        equations = self.predictor.to_symbolic()

        # If the output is greater than 1, equations will be a list.
        # Each equation in the list will have the same complexity, therefore we return only the first one.
        if self.output_size > 1:
            store_eq(equations[0], log_dir)
        store_eq(equations, log_dir)