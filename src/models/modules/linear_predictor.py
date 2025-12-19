import torch
import torch.nn as nn
from src.models.encoders.linear import LinearEncoder

class EquationDecoder(nn.Module):
    def __init__(self,
                 latent_size,
                 n_parameters,
                 n_classes,
                 activation='ReLU',
                 ):
        super(EquationDecoder, self).__init__()
        self.latent_size = latent_size
        self.n_parameters = n_parameters
        self.n_classes = n_classes

        self.decoder = LinearEncoder(
            input_size=latent_size,
            output_size=n_classes * n_parameters,
            activation=activation,
        )

    def forward(self, memory):
        # Get the parameters of the linear equations stored in memory.
        equation_weights = self.decoder(memory)

        # Reshape to: (memory_size, parameters, n_classes)
        equation_weights = equation_weights.view(-1, self.n_parameters, self.n_classes)

        return equation_weights


class LinearPredictor(nn.Module):
    def __init__(self,
                 memory_size, 
                 latent_size,
                 c_names,
                 y_names,
                 bias='global',
                 activation='ReLU',
                 ):
        super(LinearPredictor, self).__init__()

        self.memory_size = memory_size
        self.latent_size = latent_size
        self.c_names = c_names
        self.bias = bias
        self.y_names = y_names

        # check bias
        if self.bias not in ['global', 'local', None]:
            raise ValueError(f"Unknown bias type: {self.bias}")

        # The memory contains a set of linear predictors.
        self.parameters = self.c_names + ['bias'] if self.bias == 'local' else self.c_names
        self.equation_memory = torch.nn.Embedding(
            memory_size,
            latent_size
        )
        self.equation_decoder = EquationDecoder(
            latent_size=latent_size,
            n_classes=len(self.parameters),
            n_parameters=len(y_names),
            activation=activation,
        )

        # Global bias
        if self.bias == 'global':
            self.bias_params = nn.Parameter(torch.zeros(len(self.y_names))) 

    def forward(self, prob_per_classifier, input_concepts):
        bsz = prob_per_classifier.shape[0]
        # Get the parameters of the linear equations stored in memory.
        equation_weights = self.equation_decoder(self.equation_memory.weight)

        # Reshape to: (memory_size, parameters, n_classes)
        equation_weights = equation_weights.view(self.memory_size, len(self.parameters), len(self.y_names))

        # Adding batch dimension to concept memory
        equation_weights = equation_weights.unsqueeze(dim=0).expand(bsz, -1, -1, -1)

        # Get the weights to generate the explanation
        predicted_weights = self.get_weights_for_explanation(equation_weights, 
                                                             prob_per_classifier)

        # Execute the linear equations stored in memory by performing the dot product 
        # among the input concepts and the weights of the linear equations.
        # Dimension: (batch_size, output_size, memory_size)
        y_per_classifier = self.linear_equation_eval(equation_weights, input_concepts)
        
        # Select one logit for each class of y form the memory
        # Dimension: (batch_size, output_size, n_samples)
        y_hat = self.selection_eval(prob_per_classifier, y_per_classifier)
        if self.bias=='global':
            y_hat = y_hat + self.bias_params[None, :, None]

        return {
            'y_hat': y_hat,
            'explanations': predicted_weights,
        }


    def linear_equation_eval(self, memory, input_concepts):
        if self.bias == 'local':
            concept_memory = memory[:,:,:-1,:]
            bias_memory = memory[:,:,-1,:].permute(0, 2, 1)
        else:
            concept_memory = memory
            bias_memory = None
        
        y_hat = torch.einsum('bmcy,bc->bym', concept_memory, input_concepts)

        if bias_memory is not None:
            y_hat = y_hat + bias_memory

        return y_hat

    def selection_eval(self, prob_per_classifier, y_per_classifier):
        """
        Select the linear classifier from the memory based on the probabilities
        computed by the classifier selector.
        With independent outputs, prob_per_classifier has shape (batch_size, output_size, memory_size, n_samples)
        and y_per_classifier has shape (batch_size, output_size, memory_size).
        The output dimension is (batch_size, output_size, n_samples).
        """
        return torch.einsum('bmys,bym->bys', prob_per_classifier, y_per_classifier)
    
    def get_weights_for_explanation(self, memory, selection):
        """
        Get the classifier's weights selection according to the distribution probabilities.
        With independent outputs, selection has shape (batch_size, output_size, memory_size, n_samples).
        The output dimension is (batch_size, output_size, n_parameters, n_samples).
        """
        return torch.einsum('bmcy,bmys->bycs', memory, selection)
        
