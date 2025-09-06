import torch
from sympy import sympify, symbols
from torchmetrics import Metric
from sklearn.metrics import f1_score, accuracy_score
from torchmetrics.classification import BinaryAUROC, BinaryF1Score

class MAE(Metric):
    """
    Mean Absolute Error (MAE) metric for regression tasks.
    """
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # tasks predictions will be in the shape (batch_size, 1) containing the predicted value
        # concepts predictions will be in the shape (batch_size, n_concepts) containing the predicted value for each concept
        assert preds.shape == target.shape
        abs_error = torch.abs(preds - target)
        self.sum_abs_error += torch.sum(abs_error)
        self.total += target.numel()

    def compute(self):
        return self.sum_abs_error / self.total


class MSE(Metric):
    """
    Mean Squared Error (MSE) metric for regression tasks.
    """
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # tasks predictions will be in the shape (batch_size, 1) containing the predicted value
        # concepts predictions will be in the shape (batch_size, n_concepts) containing the predicted value for each concept
        assert preds.shape == target.shape
        squared_error = (preds - target) ** 2
        self.sum_squared_error += torch.sum(squared_error)
        self.total += target.numel()

    def compute(self):
        return self.sum_squared_error / self.total


class ClassAccuracy(Metric):
    """
    Classification accuracy metric of the pytorch_lightning model.
    """
    def __init__(self,  
                 dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # tasks predictions will be in the shape (batch_size, 1) containing the index of the predicted class
        # concepts predictions will be in the shape (batch_size, n_concepts) containing the predicted probability for each binary concept
        assert preds.shape == target.shape
        preds = preds.long()
        target = target.long()
        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total


def f1_acc_metrics(y_true, y_pred):
    """
    Calculate the F1 score and accuracy for the given true and predicted labels.
    """
    # Convert PyTorch tensors to lists if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy().tolist()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy().tolist()
    
    # Calculate the F1 score
    f1 = f1_score(y_true, y_pred, average='macro')
    # Calculate the accuracy
    accuracy = accuracy_score(y_true, y_pred)
    return f1, accuracy


'''
class GenAccuracy(Metric):
    """
    Generation accuracy metric of the pytorch_lightning model.
    """
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # remove invalid predictions associated with padding
        mask = target[:, 0] != -100
        preds = preds[mask]
        target = target[mask]

        preds = preds.long()
        target = target.squeeze().long()

        preds = torch.argmax(preds, dim=1)
        target = torch.argmax(target, dim=-1)

        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total

class LogitBinaryAUROC(BinaryAUROC):
    """
    AUROC metric from pytorch metrics working with logits.
    """
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = torch.sigmoid(preds)
        super().update(preds, target)

class LogitBinaryF1Score(BinaryF1Score):
    """
    F1 score metric from pytorch metrics working with logits.
    """
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = torch.sigmoid(preds)
        super().update(preds, target)
'''



""""""
if __name__ == '__main__':
    metric = ClassAccuracy()
    output = torch.tensor([[1,0,1],
                          [0,0,1],
                          [1,0,1]])  # Example output
    label = torch.tensor([[1,0,1],
                          [0,0,1],
                          [1,0,1]])          # Example label
    metric.update(output, label)
    print("Class accuracy:", metric.compute().item())

    # --- Define expressions ---
    from utils.ted import make_costs, sympy_to_tree, ted_weighted, ted_weighted_normalized
    from sympy import symbols, sin, cos, log, Rational

    c0, c1 = symbols('c0 c1')

    f_true      = 2*sin(c0) + c1**2 + Rational(1, 2)
    f_learned_1 = 2*sin(c0) + c1*c1  + Rational(1, 2)   # algebraically equal, structurally different (Pow vs Mul)
    f_learned_2 = 2*sin(c0) + c1**2 + Rational(50001, 100000) # tiny bias change
    f_learned_3 =   sin(c0) + c1**2 - Rational(1, 10) # different scale and bias
    f_learned_4 = 2*cos(c0) + c1**2 + Rational(1, 2) # (sin -> cos)
    f_learned_5 = 2*sin(c0) + c1 + Rational(1, 2) # (c1^2 -> c1) 
    f_learned_6 = 2*cos(c0)**2 + log(c0) + c0 - Rational(1, 5) # large difference

    T_true = sympy_to_tree(f_true, canonicalize_commutative=True)
    print("True:    ", f_true)
    
    """
    How to tune it
    --------------

    - Bias sensitivity
    Lower weight_num_leaf (e.g., 0.1–0.3) to make adding/removing a constant (bias) cheap. Set higher if you want bias edits to count more.

    - Coefficient tolerance
    Increase num_tol_rel (e.g., 1e-2) to ignore small relative coefficient changes. The replacement cost scales with relative difference and is capped by num_replace_cap.

    - Global scale/shift invariance
    Turn Option B on (the affine alignment step). Choose an input grid that reflects your domain of interest; in higher dimensions, pass multiple variables via symbols_order and add them to grid.

    - Operator vs. numeric emphasis
    Change op_rename_cost and sym_rename_cost to emphasize structural edits over numeric tweaks (or vice versa).
    """
    node_weight, rename_cost = make_costs(
        # Treat small relative coefficient differences as “no change”
        num_tol_abs=1e-8,      # tiny absolute floor
        num_tol_rel=1e-2,      # 1% relative tolerance (good for ~1e-3…1e3 range)

        # Even when numbers differ beyond tolerance, keep the penalty small
        num_replace_cap=0.20,  # at most 0.20 cost to change any numeric leaf
        num_replace_scale=0.50,# slope toward the cap

        # Structural renames dominate numeric/symbolic
        op_rename_cost=1.5,    # changing +↔*, sin↔cos, etc. is expensive
        sym_rename_cost=0.5,   # renaming a variable is cheaper than changing ops

        # Insertion/deletion weights (subtree costs are sums of these)
        weight_num_leaf=0.05,  # constants/bias terms are very cheap to add/remove
        weight_sym_leaf=0.60,  # variables moderate
        weight_op_node=1.30,   # operators heavier ⇒ structure matters most
    )

    for f in [f_learned_1, f_learned_2, f_learned_3, f_learned_4, f_learned_5, f_learned_6]:
        print(f"Learned: {f} \t", end=' ')
        T = sympy_to_tree(f, canonicalize_commutative=True)
        d_raw = ted_weighted(T_true, T, node_weight, rename_cost)
        dn_raw = ted_weighted_normalized(T_true, T, node_weight, rename_cost, mode="max")
        print(f"TED = {d_raw:.3f}, normalized = {dn_raw:.3f}")