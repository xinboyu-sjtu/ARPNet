#!/usr/bin/env python
"""Run ARPNet inference on a manifest.

Supports both the representative multimodal model and the image-only control
model.  The multimodal model requires ``--clinical_csv``; the image-only model
does not use clinical variables.

Example
-------
python scripts/predict.py \
    --model_type multimodal \
    --manifest examples/manifest_example.csv \
    --clinical_csv examples/clinical_example.csv \
    --split test \
    --stage1_ckpt runs/stage1/best.pth \
    --stage2_ckpt runs/stage2/best.pth \
    --out_dir predictions \
    --eval_stage1

python scripts/predict.py \
    --model_type image_only \
    --manifest examples/manifest_example.csv \
    --split test \
    --stage1_ckpt runs/stage1/best.pth \
    --stage2_ckpt checkpoints/ARPNetStageII.pth \
    --out_dir predictions_image_only
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from arpnet.data.clinical import (
    CLINICAL_DIM,
    MultimodalDataset,
    load_clinical_csv,
)
from arpnet.data.dataset import PairedCBCTDataset
from arpnet.data.preprocessing import keep_largest_component, write_image
from arpnet.metrics import segmentation_scores
from arpnet.models.arpnet import ARPNet
from arpnet.models.stage1 import ARPNetStageI
from arpnet.models.stage2 import ARPNetStageII, ARPNetStageIIImageOnly
from arpnet.utils import get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ARPNet inference.")
    parser.add_argument(
        "--model_type",
        choices=("multimodal", "image_only"),
        default="multimodal",
        help="Representative multimodal model or image-only control model.",
    )
    parser.add_argument("--manifest", required=True, help="CSV manifest.")
    parser.add_argument(
        "--clinical_csv",
        default=None,
        help=(
            "Clinical CSV with case_id, tooth_type, jaw, age and gender. "
            "Required for --model_type multimodal."
        ),
    )
    parser.add_argument("--split", default=None, help="Optional split to evaluate.")
    parser.add_argument("--stage1_ckpt", required=True, help="Stage I checkpoint.")
    parser.add_argument("--stage2_ckpt", required=True, help="Stage II checkpoint.")
    parser.add_argument("--out_dir", default="predictions", help="Output directory.")
    parser.add_argument("--image_size", type=int, default=192, help="Cubic volume size.")
    parser.add_argument("--down_scale", type=int, default=3, help="Stage I scale branches.")
    parser.add_argument(
        "--stage1_loss",
        choices=("mse", "dice", "bce"),
        default="mse",
        help="Stage I loss; Dice/BCE apply a sigmoid before thresholding.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Mask threshold.")
    parser.add_argument(
        "--eval_stage1",
        action="store_true",
        help="Load the postoperative ground truth and compute Stage I metrics.",
    )
    parser.add_argument(
        "--save_masks",
        action="store_true",
        help="Write one predicted NIfTI mask per case.",
    )
    parser.add_argument(
        "--keep_largest",
        action="store_true",
        help="Keep only the largest connected component of each predicted mask.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default=None, help="Device, e.g. cuda:0 or cpu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model_type == "multimodal" and not args.clinical_csv:
        raise ValueError(
            "--clinical_csv is required when --model_type multimodal."
        )

    os.makedirs(args.out_dir, exist_ok=True)
    mask_dir = os.path.join(args.out_dir, "masks")
    if args.save_masks:
        os.makedirs(mask_dir, exist_ok=True)
    set_seed(args.seed)

    device = get_device(args.device)

    base_dataset = PairedCBCTDataset(
        manifest=args.manifest,
        split=args.split,
        image_size=args.image_size,
        require_label=False,
        load_post=args.eval_stage1,
    )

    # Stage I
    stage1 = ARPNetStageI(
        image_size=args.image_size,
        down_scale=args.down_scale,
    ).to(device)
    stage1_payload = torch.load(args.stage1_ckpt, map_location=device)
    stage1_loss = args.stage1_loss
    if isinstance(stage1_payload, dict) and "state_dict" in stage1_payload:
        stage1_loss = stage1_payload.get("loss", stage1_loss)
        stage1_payload = stage1_payload["state_dict"]
    stage1.load_state_dict(stage1_payload)
    stage1.eval()

    # Stage II
    stage2_payload = torch.load(args.stage2_ckpt, map_location=device)
    stage2_meta = stage2_payload if isinstance(stage2_payload, dict) else {}
    if "state_dict" in stage2_meta:
        stage2_state = stage2_meta["state_dict"]
    else:
        stage2_state = stage2_payload

    clinical_data = {}
    if args.model_type == "multimodal":
        clinical_data, _, _ = load_clinical_csv(
            args.clinical_csv,
            age_mean=stage2_meta.get("clinical_age_mean"),
            age_std=stage2_meta.get("clinical_age_std"),
        )
        dataset = MultimodalDataset(base_dataset, clinical_data)
        stage2 = ARPNetStageII(
            num_classes=3,
            in_channels=1,
            clinical_dim=CLINICAL_DIM,
        ).to(device)
        stage2.load_state_dict(stage2_state)
        stage2.eval()
        model = ARPNet(
            stage1=stage1,
            stage2=stage2,
            stage1_use_sigmoid=(stage1_loss != "mse"),
            mask_threshold=args.threshold,
        ).to(device)
        model.eval()
    else:
        dataset = base_dataset
        stage2 = ARPNetStageIIImageOnly(
            num_classes=3,
            in_channels=1,
        ).to(device)
        stage2.load_state_dict(stage2_state)
        stage2.eval()
        model = None

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    rows = []
    global_index = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            if args.model_type == "multimodal":
                pre, post, label, clinical, case_ids = batch
                pre = pre.to(device, non_blocking=True)
                clinical = clinical.to(device, non_blocking=True)
                logits, mask = model(pre, clinical, return_mask=True)
            else:
                pre, post, label, case_ids = batch
                pre = pre.to(device, non_blocking=True)
                stage1_out = stage1(pre)
                if stage1_loss != "mse":
                    stage1_out = torch.sigmoid(stage1_out)
                mask = (stage1_out >= args.threshold).float()
                logits = stage2(pre, mask)

            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            predictions = probabilities.argmax(axis=1)
            mask = mask.cpu().numpy()

            for batch_idx, case_id in enumerate(case_ids):
                pred_mask = mask[batch_idx, 0]
                if args.keep_largest:
                    pred_mask = keep_largest_component(pred_mask)

                score = {}
                if args.eval_stage1:
                    target = post[batch_idx, 0].cpu().numpy()
                    score = segmentation_scores(pred_mask, target)

                row = {
                    "case_id": case_id,
                    "model_type": args.model_type,
                    "clinical_available": int(
                        str(case_id) in clinical_data
                    ),
                    "label": int(label[batch_idx]),
                    "pred_label": int(predictions[batch_idx]),
                    "prob_0": float(probabilities[batch_idx, 0]),
                    "prob_1": float(probabilities[batch_idx, 1]),
                    "prob_2": float(probabilities[batch_idx, 2]),
                }
                row.update(
                    {
                        "dice": score.get("dice", np.nan),
                        "iou": score.get("iou", np.nan),
                        "precision": score.get("precision", np.nan),
                        "recall": score.get("recall", np.nan),
                        "intersection": score.get("intersection", np.nan),
                        "pred_voxels": score.get("pred_voxels", np.nan),
                        "gt_voxels": score.get("gt_voxels", np.nan),
                    }
                )
                rows.append(row)

                if args.save_masks:
                    reference = sitk.ReadImage(
                        base_dataset.pre_path_at(global_index)
                    )
                    write_image(
                        pred_mask,
                        reference,
                        os.path.join(mask_dir, f"{case_id}_pred.nii.gz"),
                    )
                global_index += 1

    frame = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "predictions.csv")
    frame.to_csv(csv_path, index=False)
    print(f"Saved predictions -> {csv_path}")
    if args.save_masks:
        print(f"Saved masks -> {mask_dir}")


if __name__ == "__main__":
    main()
