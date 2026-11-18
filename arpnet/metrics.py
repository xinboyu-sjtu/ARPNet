"""Evaluation metrics for Stage I segmentation and Stage II classification."""

from typing import Dict, Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def segmentation_scores(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Per-case Dice, IoU, precision and recall for binary masks.

    ``pred`` and ``target`` are binary arrays of the same shape.  The returned
    counts are useful for computing pooled (voxel-micro) metrics later.
    """
    pred = (pred >= 0.5).astype(np.float64)
    target = (target >= 0.5).astype(np.float64)

    intersection = float((pred * target).sum())
    pred_sum = float(pred.sum())
    target_sum = float(target.sum())
    union = pred_sum + target_sum - intersection

    return {
        "dice": (2.0 * intersection + eps) / (pred_sum + target_sum + eps),
        "iou": (intersection + eps) / (union + eps),
        "precision": (intersection + eps) / (pred_sum + eps),
        "recall": (intersection + eps) / (target_sum + eps),
        "intersection": intersection,
        "pred_voxels": pred_sum,
        "gt_voxels": target_sum,
    }


def dice_score_tensor(
    pred: torch.Tensor,
    target: torch.Tensor,
    from_logits: bool = False,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """Mean hard Dice over a batch of binary masks.

    ``pred`` and ``target`` have shape ``[B, 1, D, H, W]`` (or any shape whose
    first dimension is the batch dimension).  If ``from_logits`` is True, a
    sigmoid is applied before thresholding.
    """
    if from_logits:
        pred = torch.sigmoid(pred)
    pred = (pred >= threshold).float()
    target = (target >= 0.5).float()

    batch = pred.shape[0]
    pred = pred.reshape(batch, -1)
    target = target.reshape(batch, -1)
    numerator = 2.0 * (pred * target).sum(dim=1) + eps
    denominator = pred.sum(dim=1) + target.sum(dim=1) + eps
    return float((numerator / denominator).mean().item())


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    """Return mean and standard deviation, ignoring NaNs."""
    array = np.asarray(values, dtype=np.float64)
    array = array[~np.isnan(array)]
    if array.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "n": int(array.size),
    }


def pooled_segmentation(
    intersection: Sequence[float],
    pred_voxels: Sequence[float],
    gt_voxels: Sequence[float],
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Voxel-micro Dice and IoU computed from summed counts."""
    inter = float(np.sum(intersection))
    pred = float(np.sum(pred_voxels))
    gt = float(np.sum(gt_voxels))
    return {
        "dice": (2.0 * inter + eps) / (pred + gt + eps),
        "iou": (inter + eps) / (pred + gt - inter + eps),
    }


def calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 5,
) -> Sequence[Dict[str, float]]:
    """Micro-averaged reliability curve with equal-width bins.

    All class probabilities are flattened together with their one-hot targets,
    so each bin summarises every (case, class) prediction.  Returns one record
    per non-empty bin with the bin boundaries, count, mean predicted
    probability and observed positive fraction.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    row_sums = y_prob.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Each probability row must have a positive sum.")
    y_prob = y_prob / row_sums
    targets = np.eye(y_prob.shape[1])[y_true].reshape(-1)
    probabilities = y_prob.reshape(-1)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probabilities, edges[1:-1], right=True)
    rows = []
    for bin_index in range(n_bins):
        mask = bin_ids == bin_index
        if not mask.any():
            continue
        rows.append(
            {
                "bin": int(bin_index),
                "bin_low": float(edges[bin_index]),
                "bin_high": float(edges[bin_index + 1]),
                "count": int(mask.sum()),
                "mean_pred": float(probabilities[mask].mean()),
                "frac_pos": float(targets[mask].mean()),
            }
        )
    return rows


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 5,
) -> float:
    """Micro-averaged expected calibration error (ECE)."""
    rows = calibration_curve(y_true, y_prob, n_bins=n_bins)
    total = sum(row["count"] for row in rows)
    if total == 0:
        return float("nan")
    return float(
        sum(
            row["count"] / total * abs(row["frac_pos"] - row["mean_pred"])
            for row in rows
        )
    )


def brier_skill_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    train_priors: Sequence[float],
) -> Dict[str, float]:
    """Brier score, no-information reference and Brier skill score.

    The reference model predicts the training-set class priors for every case.
    Both Brier scores use the paper's per-element normalization, i.e. the mean
    over cases and classes.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    row_sums = y_prob.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Each probability row must have a positive sum.")
    y_prob = y_prob / row_sums
    priors = np.asarray(train_priors, dtype=np.float64)
    if priors.size != y_prob.shape[1]:
        raise ValueError(
            "train_priors must have one value per class "
            f"({y_prob.shape[1]}), got {priors.size}."
        )
    priors = priors / priors.sum()
    one_hot = np.eye(y_prob.shape[1])[y_true]

    brier = float(np.mean((y_prob - one_hot) ** 2))
    reference = float(np.mean((priors[None, :] - one_hot) ** 2))
    skill = float(1.0 - brier / reference) if reference > 0 else float("nan")
    return {
        "brier": brier,
        "brier_reference": reference,
        "bss": skill,
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int = 3,
    train_priors: Optional[Sequence[float]] = None,
    n_bins: int = 5,
) -> Dict[str, float]:
    """Accuracy, weighted/macro precision-recall-F1, AUROC, Brier, ECE, BSS.

    The multi-class Brier score is averaged over both cases and classes:

        (1 / (N * C)) * sum_i sum_c (p_ic - y_ic)^2

    This is the definition used in the paper.  Weighted precision/recall/F1 use
    the test-set class support, so weighted recall equals accuracy for
    single-label classification.  If ``train_priors`` is given, the
    no-information Brier reference and the Brier skill score are also returned.
    If a class is absent from ``y_true`` (common in small external cohorts),
    AUROC is computed over the classes that are present.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_prob.ndim != 2 or y_prob.shape[1] != num_classes:
        raise ValueError(
            f"y_prob must have shape (N, {num_classes}), got {y_prob.shape}."
        )
    row_sums = y_prob.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Each probability row must have a positive sum.")
    y_prob = y_prob / row_sums

    result = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }

    present = np.unique(y_true)
    aucs = []
    for class_index in present:
        binary_target = (y_true == class_index).astype(int)
        if binary_target.min() == binary_target.max():
            continue
        try:
            aucs.append(
                roc_auc_score(binary_target, y_prob[:, class_index])
            )
        except ValueError:
            continue
    result["auroc_macro"] = float(np.mean(aucs)) if aucs else float("nan")

    one_hot = np.eye(num_classes)[y_true]
    # Average over both cases and classes, i.e. 1/(N*C) * sum_i sum_c.
    result["brier"] = float(np.mean((y_prob - one_hot) ** 2))
    result["ece"] = expected_calibration_error(
        y_true, y_prob, n_bins=n_bins
    )

    if train_priors is not None:
        result.update(brier_skill_score(y_true, y_prob, train_priors))
    return result


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Percentile bootstrap confidence interval for the mean of ``values``."""
    array = np.asarray(values, dtype=np.float64)
    array = array[~np.isnan(array)]
    if array.size == 0 or n_boot <= 0:
        return {
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n": int(array.size),
        }

    rng = np.random.default_rng(seed)
    n = array.size
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        means[i] = array[rng.integers(0, n, n)].mean()

    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(means, alpha / 2)),
        "ci_high": float(np.quantile(means, 1 - alpha / 2)),
        "n": int(array.size),
    }


def bootstrap_classification_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "accuracy",
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    num_classes: int = 3,
    train_priors: Optional[Sequence[float]] = None,
    n_bins: int = 5,
) -> Dict[str, float]:
    """Percentile bootstrap CI for one classification metric."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    values = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        indices = rng.integers(0, n, n)
        metrics = classification_metrics(
            y_true[indices],
            y_pred[indices],
            y_prob[indices],
            num_classes=num_classes,
            train_priors=train_priors,
            n_bins=n_bins,
        )
        values[i] = metrics.get(metric, np.nan)

    valid = values[~np.isnan(values)]
    return {
        "mean": float(np.nanmean(values)),
        "ci_low": float(np.nanquantile(values, alpha / 2)),
        "ci_high": float(np.nanquantile(values, 1 - alpha / 2)),
        "n_valid": int(valid.size),
    }
