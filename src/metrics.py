import torch
from torchmetrics import Metric
from sklearn.metrics import f1_score, accuracy_score

class Task_Accuracy(Metric):
    """
    Task accuracy metric of the pytorch_lightning model.
    """
    def __init__(self, logic_reasoning=False, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.logic_reasoning = logic_reasoning

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if len(preds.squeeze().shape) > 1:
            preds = torch.argmax(preds, dim=1)
        else:
            if self.logic_reasoning:
                preds = preds.squeeze() > 0.5
            else:
                preds = preds.squeeze() > 0. 
        target = target.squeeze()
        assert preds.shape == target.shape
        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total


class Concept_Accuracy(Metric):
    """
    Concept accuracy metric of the pytorch_lightning model.
    """
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = torch.where(preds > 0.5, 1, 0)
        assert preds.shape == target.shape
        self.correct += torch.sum(preds == target)
        self.total += target.shape[0] * target.shape[1]

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
