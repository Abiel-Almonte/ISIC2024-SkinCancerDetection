import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from sklearn.metrics import roc_auc_score
from pytorch_metric_learning import losses
from typing import Dict, Type, Protocol

__all__= ['BCELoss', 'FocalLoss', 'BCEVAELoss', 'ContrastiveLoss', 'LossFn', 'get_lossfn', 'register_lossfn']

class LossFn(Protocol):
    def forward(self, *args, **kwargs) -> torch.Tensor:
        ...

loss_registry:  Dict[str, Type[LossFn]] = {}

class BCELoss(nn.BCEWithLogitsLoss, LossFn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(input, target)

class FocalLoss(nn.Module, LossFn):
    def __init__(self, alpha=0.25, gamma=2)-> None:
        """
        Initializes the Focal Loss.

        Parameters:
            alpha (float): Weighting factor for the rare class.
            gamma (float): Focusing parameter to adjust the rate at which easy examples are down-weighted.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor)-> torch.Tensor:
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        return F_loss.mean()
    
class BCEVAELoss(nn.Module, LossFn):
    def __init__(self)-> None:
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor)->torch.Tensor:
        """
        Computes the Binary Cross-Entropy and KL Divergence Loss.

        Parameters:
            inputs (torch.Tensor): The predicted values.
            targets (torch.Tensor): The ground truth values.
            mu (torch.Tensor): The mean of the latent distribution.
            logvar (torch.Tensor): The log variance of the latent distribution.

        Returns:
            torch.Tensor: The combined BCE and KL loss.
        """
        bce_loss= F.binary_cross_entropy_with_logits(inputs, targets)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return bce_loss+ kl_loss
    
def partial_auc(labels, predictions, min_tpr: float=0.80):
    """
    Computes the Partial AUC score.

    Parameters:
        labels (np.ndarray): The true binary labels.
        predictions (np.ndarray): The predicted scores.
        min_tpr (float): The minimum true positive rate for the partial AUC calculation.

    Returns:
        float: The computed partial AUC score.
    """
    labels= numpy.asarray(labels)
    predictions= numpy.asarray(predictions)

    labels= numpy.abs(labels - 1)
    predictions= 1.0 - predictions

    max_fpr= 1-min_tpr
    partial_auc_scaled= roc_auc_score(labels, predictions, max_fpr=max_fpr)

    return 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)

def register_lossfn(name: str, LossClass: Type[LossFn]):
    loss_registry[name] = LossClass

def get_lossfn(name: str):
    if name not in loss_registry:
        raise ValueError(f"Model {name} not found. Choose one of the following: {list(loss_registry.keys())}")
    return loss_registry[name]


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.3):
        super().__init__()
        self.temperature = temperature

    def forward(self, feature_vectors, labels):
        # Normalize feature vectors
        feature_vectors_normalized = F.normalize(feature_vectors, p=2, dim=1)
        # Compute logits
        logits = torch.div(
            torch.matmul(
                feature_vectors_normalized, torch.transpose(feature_vectors_normalized, 0, 1)
            ),
            self.temperature,
        )
        return losses.NTXentLoss(temperature=0.07)(logits, torch.squeeze(labels))