import torch
import torch.nn as nn
import torch_concepts.nn as pyc_nn
from torch_concepts.nn import functional as CF
from src.models.base import BaseModel, LogicModel
from torch.nn import functional as F

class ConceptMemoryReasoner(LogicModel):
    def __init__(self, 
                 input_size, 
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
                 conc_rec_weight=1.0
                 ):
        super().__init__(
            input_size,
            output_size,
            task,
            activation,
            latent_size,
            c_groups
        )

        self.n_roles = 3
        self.memory_names = ['Positive', 'Negative', 'Irrelevant']
        
        self.embedding_size = embedding_size
        self.latent_size = latent_size
        self.task_penalty = task_penalty
        self.c_names = list(c_names)
        self.int_prob = int_prob
        self.int_idxs = int_idxs
        self.has_concepts = True
        self.noise = noise
        self.y_names = list(y_names)
        self._multi_class = len(self.y_names) > 1

        self.memory_size = memory_size
        self.rec_weight = conc_rec_weight

        self.bottleneck = pyc_nn.LinearConceptBottleneck(
            latent_size,
            self.c_names,
        )

        self.concept_memory = torch.nn.Embedding(
            memory_size,
            latent_size,
        )
        self.memory_decoder = pyc_nn.LinearConceptLayer(
            latent_size,
            [
                self.c_names,
                self.y_names,
                self.memory_names,
            ],
        )
        self.classifier_selector = nn.Sequential(
            pyc_nn.LinearConceptLayer(
                latent_size,
                [self.y_names, memory_size],
            ),
        )

        self.concept_loss_form = nn.BCELoss()
        self.task_loss_form = nn.BCELoss()


    def _conc_recon(self, concept_weights, c_true, y_true):
        # check if y_true is an array (label encoding) or a matrix
        # (one-hot encoding) in case it is an array convert it to a matrix
        # if it is a multi-class task
        if len(y_true.squeeze().shape) == 1 and self._multi_class:
            y_true = torch.nn.functional.one_hot(
                y_true.squeeze().long(),
                len(self.y_names),
            )

        elif len(y_true.shape) == 1:
            y_true = y_true.unsqueeze(-1)
        c_rec_per_classifier = CF.logic_memory_reconstruction(
            concept_weights,
            c_true,
            y_true,
        )
        # weighting the reconstruction loss - lower reconstruction weights
        # brings values closer to 1 thus influencing less the prediction
        c_rec_per_classifier = torch.pow(c_rec_per_classifier, self.rec_weight)

        return c_rec_per_classifier

    def forward(self, input):
        latent, c_true, int_idxs = self.encode(input)

        y_true = input['y'] if self.training else None

        c_emb, c_dict = self.bottleneck(
            latent,
            c_true=c_true,
            intervention_idxs=int_idxs,
            intervention_rate=1.,
        )
        c_pred = c_dict['c_int']
        classifier_selector_logits = self.classifier_selector(latent)
        prob_per_classifier = torch.softmax(classifier_selector_logits, dim=-1)
        # softmax over roles and adding batch dimension to concept memory
        concept_weights = self.memory_decoder(
            self.concept_memory.weight).softmax(dim=-1).unsqueeze(dim=0)

        c_input = c_pred > 0.5 if self.hard_concepts else c_pred
        y_per_classifier = CF.logic_rule_eval(concept_weights, c_input)

        if y_true is not None:
            c_rec_per_classifier = self._conc_recon(concept_weights,
                                                    c_true,
                                                    y_true)
            y_pred = CF.selection_eval(
                prob_per_classifier,
                y_per_classifier,
                c_rec_per_classifier,
            )
        else:
            y_pred = CF.selection_eval(prob_per_classifier,
                                       y_per_classifier)
            
        return y_pred, c_pred
    
    def loss(self, y_hat, y, c_hat=None, c=None):
        # one hot encode the y variable
        if self.task == 'classification' and self.output_size > 1:
            y = F.one_hot(y.flatten().long(), num_classes=self.output_size).float()
        loss = self.concept_based_loss(y_hat, y, c_hat, c)
        return loss


