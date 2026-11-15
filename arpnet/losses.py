"""Loss functions used by ARPNet."""

import torch
import torch.nn.functional as F


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Voxel-wise mean squared error.

    For Stage I the model output is a raw mask; for Stage II the output is a
    probability vector and the target is a one-hot label (multi-class Brier
    score).
    """
    return F.mse_loss(pred, target)


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1.0,
) -> torch.Tensor:
    """Soft Dice loss on ``sigmoid(logits)``, averaged over the batch."""
    probabilities = torch.sigmoid(logits)
    numerator = 2.0 * (probabilities * target).sum(dim=(1, 2, 3, 4)) + eps
    denominator = (
        probabilities.sum(dim=(1, 2, 3, 4))
        + target.sum(dim=(1, 2, 3, 4))
        + eps
    )
    return (1.0 - numerator / denominator).mean()


def bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy with logits."""
    return F.binary_cross_entropy_with_logits(logits, target)


def multiclass_brier(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Multi-class Brier score (MSE between softmax probabilities and one-hot)."""
    probabilities = F.softmax(logits, dim=1)
    target_one_hot = F.one_hot(target, num_classes=logits.shape[1]).float()
    return F.mse_loss(probabilities, target_one_hot)


def weighted_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor = None,
) -> torch.Tensor:
    """Cross-entropy with optional per-class weights."""
    return F.cross_entropy(logits, target, weight=class_weights)
