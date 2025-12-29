import torch
import torch.nn as nn
import torch_concepts.nn as pyc_nn
from torch_concepts.nn import functional as CF
from torch_concepts.semantic import CMRSemantic
from torch_concepts.nn import concept_embedding_mixture
from src.utils.expression_utils import boolean_and_expression, store_eq
from src.models.baselines.base import BaseModel
from torch.nn import functional as F

eps = 1e-8

class ConceptMemoryReasoner(BaseModel):
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
                 conc_rec_weight=1.0,
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
        
        self.embedding_size = embedding_size
        self.task_penalty = task_penalty * 3 # BCE gives lower loss values
        self.has_concepts = True
        self.y_names = list(y_names)
        self._multi_class = len(self.y_names) > 1

        self.memory_size = memory_size
        self.rec_weight = conc_rec_weight

        self.bottleneck = pyc_nn.LinearConceptBottleneck(
            backbone_latent_size,
            self.c_names,
            activation=nn.Identity(),
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
                backbone_latent_size,
                [self.y_names, memory_size],
            ),
        )

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
        c_rec_per_classifier = (eps / 2 + c_rec_per_classifier * (1 - eps/2))
        c_rec_per_classifier = torch.pow(c_rec_per_classifier, self.rec_weight)

        return c_rec_per_classifier

    def forward(self, input):
        latent, x_concepts, c_true, int_idxs = self.encode(input)

        _, c_dict = self.bottleneck(
            x_concepts,
            c_true=c_true,
            intervention_idxs=int_idxs,
            intervention_rate=1.,
        )
        c_hat = c_dict['c_int']

        c_hat, input_concepts = self._process_concepts(c_hat, c_true, int_idxs)

        classifier_selector_logits = self.classifier_selector(latent)
        prob_per_classifier = torch.softmax(classifier_selector_logits, dim=-1)
        # softmax over roles and adding batch dimension to concept memory
        concept_weights = self.memory_decoder(
            self.concept_memory.weight)

        # check
        concept_weights = (eps / 2 + concept_weights * (1 - eps/2)).softmax(dim=-1).unsqueeze(dim=0)

        y_per_classifier = self.logic_rule_eval(concept_weights, input_concepts)

        y_true = input['y'] if self.training else None

        if y_true is not None:
            c_rec_per_classifier = self._conc_recon(concept_weights,
                                                    c_true,
                                                    y_true)
            y_hat = CF.selection_eval(
                prob_per_classifier,
                y_per_classifier,
                c_rec_per_classifier,
            )
        else:
            y_hat = CF.selection_eval(prob_per_classifier,
                                       y_per_classifier)

        return {
            'y_hat': y_hat,
            'c_hat': c_hat,
            'concept_weights': concept_weights,
            'prob_per_classifier': prob_per_classifier,
        }

    def loss(self, y_hat, y, c_hat=None, c=None, *args, **kwargs):
        loss = self.concept_based_loss(y_hat, y, c_hat, c)
        if torch.isnan(loss):
            raise ValueError("Loss is NaN. Check your model and data.")
        return loss


    def logic_rule_eval(
            self,
            concept_weights: torch.Tensor,
            c_hat: torch.Tensor,
            memory_idxs: torch.Tensor = None,
            semantic=CMRSemantic()
    ) -> torch.Tensor:
        """
        Use concept weights to make predictions based on logic rules.

        Args:
            concept_weights: concept weights with shape (batch_size,
                memory_size, n_concepts, n_tasks, n_roles) with n_roles=3.
            c_hat: concept predictions with shape (batch_size, n_concepts).
            memory_idxs: Indices of rules to evaluate with shape (batch_size,
                n_tasks). Default is None (evaluate all).
            semantic: Semantic function to use for rule evaluation.

        Returns:
            torch.Tensor: Rule predictions with shape (batch_size, n_tasks,
                memory_size)
        """

        assert len(concept_weights.shape) == 5, \
            ("Size error, concept weights should be batch_size x memory_size "
             f"x n_concepts x n_tasks x n_roles. Received {concept_weights.shape}")
        memory_size = concept_weights.size(1)
        n_tasks = concept_weights.size(3)


        pos_polarity, neg_polarity, irrelevance = (
            concept_weights[..., 0],
            concept_weights[..., 1],
            concept_weights[..., 2],
        )

        if memory_idxs is None:
            # cast all to (batch_size, memory_size, n_concepts, n_tasks)
            x = c_hat.unsqueeze(1).unsqueeze(-1).expand(
                -1,
                memory_size,
                -1,
                n_tasks,
            )
        else:  # cast all to (batch_size, memory_size=1, n_concepts, n_tasks)
            # TODO: memory_idxs never used!
            x = c_hat.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, n_tasks)

        # batch_size, mem_size, n_tasks
        y_per_rule = semantic.disj(
            irrelevance,
            semantic.conj((1 - x), neg_polarity),
            semantic.conj(x, pos_polarity)
        )

        y_per_rule = (eps / 2 + y_per_rule * (1 - eps/2))

        y_per_rule = y_per_rule.clamp(min=0.0+eps, max=1.0-eps)

        assert (y_per_rule <= 1.0).all(), "y_per_rule should be in [0, 1]"
        assert (y_per_rule >= 0.0).all(), "y_per_rule should be in [0, 1]"

        # performing a conj while iterating over concepts of y_per_rule
        y_per_rule = semantic.conj(
            *[y for y in y_per_rule.split(1, dim=2)]
        ).squeeze(dim=2)

        return y_per_rule.permute(0, 2, 1)

    def get_symbolic_equivalent(self, log_dir=None):
        """
        Returns the equation associated to the predictor of the model
        """
        
        # Return the most complex linear equation that can obtained after training 
        # (all concepts are relevant and negated) 
        equation = boolean_and_expression(len(self.c_names))
        store_eq(equation, log_dir)

    def get_local_explanations(self, x, multi_label=False, **kwargs):
        latent = self.encoder(x)
        c_emb, c_dict = self.bottleneck(latent)
        c_pred = c_dict["c_int"]
        classifier_selector_logits = self.classifier_selector(latent)
        prob_per_classifier = torch.softmax(classifier_selector_logits, dim=-1)
        concept_weights = (
            self.memory_decoder(self.concept_memory.weight)
            .softmax(dim=-1)
            .unsqueeze(dim=0)
        )
        y_per_classifier = CF.logic_rule_eval(concept_weights, F.sigmoid(c_pred))
        rule_probs = prob_per_classifier * y_per_classifier
        rule_preds = rule_probs.argmax(
            dim=-1
        )  # = CF.most_likely_expl(rule_probs, multi_label)
        global_explanations = CF.logic_rule_explanations(
            concept_weights,
            {
                1: self.c_names,
                2: self.y_names,
            },
        )
        local_expl = []
        y_pred = rule_probs.sum(dim=-1)
        for i in range(x.shape[0]):
            sample_expl = {}
            for j in range(len(self.y_names)):
                # a task is predicted if it is the most likely task or is
                # a multi-label task with probability higher than 0.5 or is
                # a binary task with probability higher than 0.5
                predicted_task = (
                    (j == y_pred[i].argmax())
                    or (multi_label and y_pred[i, j] > 0.5)
                    or (not self._multi_class and y_pred[i, j] > 0.5)
                )
                if predicted_task:
                    task_rules = global_explanations[0][self.y_names[j]]
                    predicted_rule = task_rules[f"Rule {rule_preds[i, j]}"]
                    sample_expl.update({self.y_names[j]: predicted_rule})
            local_expl.append(sample_expl)
        return local_expl