#!/usr/bin/env python
"""Aggregate ARPNet predictions into the metrics reported in the paper.

The input is the ``predictions.csv`` written by ``scripts/predict.py``.
Stage I metrics are computed from the per-case Dice/IoU/precision/recall
columns; Stage II metrics are computed from the label and probability columns.

Stage II reports both support-weighted and macro precision/recall/F1, macro
one-vs-rest AUROC, the Brier score, the Brier skill score against the training
priors, and the micro-averaged expected calibration error (ECE), together with
percentile bootstrap confidence intervals.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from arpnet.metrics import (
    bootstrap_ci,
    bootstrap_classification_ci,
    calibration_curve,
    classification_metrics,
    mean_std,
    pooled_segmentation,
)

# Training-set priors from the paper: 140/224, 62/224, 22/224.
DEFAULT_TRAIN_PRIORS = (140.0 / 224.0, 62.0 / 224.0, 22.0 / 224.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate ARPNet metrics.")
    parser.add_argument("--predictions", required=True, help="predictions.csv.")
    parser.add_argument("--out", default="metrics.json", help="Output JSON path.")
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples (0 disables).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap seed.")
    parser.add_argument(
        "--train_priors",
        type=float,
        nargs=3,
        default=DEFAULT_TRAIN_PRIORS,
        help="Training-set class priors for the Brier reference model.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=5,
        help="Number of equal-width bins for the calibration curve.",
    )
    return parser.parse_args()


def stage1_summary(frame: pd.DataFrame) -> dict:
    valid = frame.dropna(subset=["dice"])
    if len(valid) == 0:
        return {}

    summary = {"n": int(len(valid))}
    for metric in ("dice", "iou", "precision", "recall"):
        summary[metric] = mean_std(valid[metric].tolist())

    summary["pooled"] = pooled_segmentation(
        valid["intersection"].tolist(),
        valid["pred_voxels"].tolist(),
        valid["gt_voxels"].tolist(),
    )
    return summary


def stage2_summary(
    frame: pd.DataFrame,
    train_priors,
    n_bins: int,
) -> dict:
    valid = frame[frame["label"] >= 0]
    if len(valid) == 0:
        return {}

    y_true = valid["label"].to_numpy(dtype=int)
    y_pred = valid["pred_label"].to_numpy(dtype=int)
    y_prob = valid[["prob_0", "prob_1", "prob_2"]].to_numpy(dtype=np.float64)
    return classification_metrics(
        y_true,
        y_pred,
        y_prob,
        train_priors=train_priors,
        n_bins=n_bins,
    )


def json_safe(value):
    """Replace NaN/Inf values with None so the output is valid JSON."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.predictions)

    result = {
        "stage1": stage1_summary(frame),
        "stage2": stage2_summary(frame, args.train_priors, args.bins),
    }

    if "stage2" in result and result["stage2"]:
        valid = frame[frame["label"] >= 0]
        y_true = valid["label"].to_numpy(dtype=int)
        y_pred = valid["pred_label"].to_numpy(dtype=int)
        y_prob = valid[["prob_0", "prob_1", "prob_2"]].to_numpy(dtype=np.float64)
        result["stage2"]["calibration_curve"] = calibration_curve(
            y_true,
            y_prob,
            n_bins=args.bins,
        )
        result["stage2"]["train_priors"] = list(args.train_priors)
        result["stage2"]["n_bins"] = int(args.bins)

        if args.bootstrap > 0:
            result["stage2"]["bootstrap"] = {}
            metrics = (
                "accuracy",
                "balanced_accuracy",
                "precision_weighted",
                "recall_weighted",
                "f1_weighted",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "auroc_macro",
                "brier",
                "ece",
                "bss",
            )
            for metric in metrics:
                result["stage2"]["bootstrap"][metric] = bootstrap_classification_ci(
                    y_true,
                    y_pred,
                    y_prob,
                    metric=metric,
                    n_boot=args.bootstrap,
                    seed=args.seed,
                    train_priors=args.train_priors,
                    n_bins=args.bins,
                )

    if args.bootstrap > 0 and "stage1" in result and result["stage1"]:
        valid = frame.dropna(subset=["dice"])
        result["stage1"]["dice_bootstrap"] = bootstrap_ci(
            valid["dice"].tolist(),
            n_boot=args.bootstrap,
            seed=args.seed,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    result = json_safe(result)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    if "stage2" in result and result["stage2"].get("calibration_curve"):
        cal_path = os.path.join(
            os.path.dirname(os.path.abspath(args.out)),
            "calibration_curve.csv",
        )
        pd.DataFrame(result["stage2"]["calibration_curve"]).to_csv(
            cal_path, index=False
        )
        print(f"Saved calibration curve -> {cal_path}")

    print(json.dumps(result, indent=2))
    print(f"\nSaved metrics -> {args.out}")


if __name__ == "__main__":
    main()
