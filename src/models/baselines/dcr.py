import torch
import torch.nn as nn
import torch_concepts.nn as pyc_nn
from torch_concepts.semantic import ProductTNorm
from torch_concepts.nn import functional as CF
from src.models.baselines.base import BaseModel
from torch.nn import functional as F
from torch_concepts.nn import concept_embedding_mixture
from src.utils.expression_utils import boolean_and_expression, store_eq

class DeepConceptReasoner(BaseModel):
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
                 semantic = ProductTNorm(),
                 temperature = 100,
                 c_groups=None,
                 hard_concepts=False,
                 encoder=None,
                 concept_loss_form=nn.BCELoss(),
                 backbone_latent_size=None,
                 concept_type='binary',
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
        self.logic_reasoning = True
        self.n_roles = 3
        self.memory_names = ['Positive', 'Negative', 'Irrelevant']
        self.y_names = list(y_names)
        
        self.embedding_size = embedding_size
        self.task_penalty = task_penalty * 3 # BCE gives lower loss values
        self.has_concepts = True
        self.semantic = semantic
        self.temperature = temperature

        self.bottleneck = pyc_nn.ConceptEmbeddingBottleneck(
            backbone_latent_size,
            self.c_names,
            embedding_size,
            activation=nn.Identity()
        )
        self.concept_importance_predictor = nn.Sequential(
            nn.Linear(embedding_size, self.latent_size),
            getattr(nn, activation)(),
            nn.Linear(self.latent_size, output_size * self.n_roles),
            nn.Unflatten(-1, (output_size, self.n_roles)),
        )

        self.task_loss_form = nn.BCELoss()

    def forward(self, input):
        x, _, c_true, int_idxs = self.encode(input)

        c_emb, c_dict = self.bottleneck(
            x,
            c_true=c_true,
            intervention_idxs=int_idxs,
            intervention_rate=1.,
        )
        c_hat = c_dict['c_int']

        c_hat, input_concepts = self._process_concepts(c_hat, c_true, int_idxs)

        # It is necessary to compute again since 
        c_emb = self.bottleneck.linear(x)
        c_emb = concept_embedding_mixture(c_emb, input_concepts)

        c_weights = self.concept_importance_predictor(c_emb)
        # adding memory dimension
        c_weights = c_weights.unsqueeze(dim=1)
        # soft selecting concept relevance (last role) among concepts
        relevance = CF.soft_select(c_weights[:, :, :, :, -2:-1],
                                   self.temperature, -3)
        # softmax over positive/negative roles
        polarity = c_weights[:, :, :, :, :-1].softmax(-1)
        # batch_size x memory_size x n_concepts x n_tasks x n_roles
        c_weights = torch.cat([polarity, 1 - relevance], dim=-1)

        y_hat = CF.logic_rule_eval(c_weights, input_concepts,
                                    semantic=self.semantic)
        # removing memory dimension
        y_hat = y_hat[:, :, 0]
        return {
            'y_hat': y_hat,
            'c_hat': c_hat,
            'c_weights': c_weights,
        }

    def loss(self, y_hat, y, c_hat=None, c=None, *args, **kwargs):
        loss = self.concept_based_loss(y_hat, y, c_hat, c)
        return loss
    
    def get_symbolic_equivalent(self, log_dir=None):
        """
        Returns the equation associated to the predictor of the model
        """
        
        # Return the most complex linear equation that can obtained after training 
        # (all concepts are relevant and negated) 
        equation = boolean_and_expression(len(self.c_names))
        store_eq(equation, log_dir)

    def get_local_explanations(self, x, multi_label=False, **kwargs):
        assert (
            not multi_label or self._multi_class
        ), "Multi-label explanations are supported only for multi-class tasks"
        latent = self.encoder(x)
        c_emb, c_dict = self.bottleneck(latent)
        c_pred = c_dict["c_int"]
        c_weights = self.concept_importance_predictor(c_emb)
        c_weights = c_weights.unsqueeze(dim=1)  # add memory dimension
        relevance = CF.soft_select(
            c_weights[:, :, :, :, -2:-1],
            self.temperature,
            -3,
        )
        polarity = c_weights[:, :, :, :, :-1].softmax(-1)
        c_weights = torch.cat([polarity, 1 - relevance], dim=-1)
        explanations = CF.logic_rule_explanations(
            c_weights,
            {
                1: self.c_names,
                2: self.y_names,
            },
        )
        
        y_pred = CF.logic_rule_eval(c_weights, F.sigmoid(c_pred), semantic=self.semantic)[:, :, 0]

        local_explanations = []
        for i in range(x.shape[0]):
            sample_expl = {}
            for j in range(len(self.y_names)):
                # a task is predicted if it is the most likely task or is
                # a multi-label task with probability higher than 0.5 or is
                # a binary task with probability higher than 0.5
                if len(self.y_names) > 1:  
                    predicted_task = j == y_pred[i].argmax()
                else:  # binary
                    predicted_task = y_pred[i, j] > 0.5

                if predicted_task:
                    task_rules = explanations[i][self.y_names[j]]
                    predicted_rule = task_rules[f"Rule {0}"]
                    sample_expl.update({self.y_names[j]: predicted_rule})
            local_explanations.append(sample_expl)
        return local_explanations