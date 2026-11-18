#!/usr/bin/env python
"""Train Stage II of ARPNet (multimodal NBA classification).

Stage I is frozen and used to generate the postoperative mask.  Stage II
combines the preoperative volume, the generated postoperative mask and an
8-dimensional clinical vector.

Example
-------
python scripts/train_stage2.py \
    --manifest examples/manifest_example.csv \
    --clinical_csv examples/clinical_example.csv \
    --stage1_ckpt runs/stage1/best.pth \
    --out_dir runs/stage2 \
    --epochs 200 \
    --loss ce
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from arpnet.data.clinical import (
    CLINICAL_DIM,
    MultimodalDataset,
    load_clinical_csv,
)
from arpnet.data.dataset import PairedCBCTDataset
from arpnet.losses import multiclass_brier, weighted_cross_entropy
from arpnet.metrics import classification_metrics
from arpnet.models.stage1 import ARPNetStageI
from arpnet.models.stage2 import ARPNetStageII
from arpnet.utils import (
    AverageMeter,
    EarlyStopping,
    count_parameters,
    get_device,
    save_checkpoint,
    save_json,
    set_seed,
    setup_logger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train multimodal ARPNet Stage II (NBA classification)."
    )
    parser.add_argument("--manifest", required=True, help="CSV manifest.")
    parser.add_argument(
        "--clinical_csv",
        required=True,
        help="Clinical CSV with case_id, tooth_type, jaw, age and gender.",
    )
    parser.add_argument("--stage1_ckpt", required=True, help="Stage I checkpoint.")
    parser.add_argument("--out_dir", default="runs/stage2", help="Output directory.")
    parser.add_argument("--train_split", default="train", help="Training split name.")
    parser.add_argument("--val_split", default="val", help="Validation split name.")
    parser.add_argument("--image_size", type=int, default=192, help="Cubic volume size.")
    parser.add_argument("--down_scale", type=int, default=3, help="Stage I scale branches.")
    parser.add_argument(
        "--stage1_loss",
        choices=("mse", "dice", "bce"),
        default="mse",
        help="Stage I loss used to decide whether to apply a sigmoid.",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs.")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument(
        "--loss",
        choices=("mse", "ce"),
        default="mse",
        help=(
            "Stage II loss. The paper uses MSE on softmax probabilities vs "
            "one-hot labels; ce enables weighted cross-entropy for the legacy "
            "released checkpoint."
        ),
    )
    parser.add_argument(
        "--class_weights",
        type=float,
        nargs=3,
        default=(1.0, 2.0, 4.0),
        help="Class weights for --loss ce.",
    )
    parser.add_argument(
        "--clinical_hidden",
        type=int,
        default=64,
        help="Width of the clinical embedding.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=1024,
        help="Width of the first fusion layer.",
    )
    parser.add_argument(
        "--fusion_dim",
        type=int,
        default=512,
        help="Width of the second fusion layer.",
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Fusion dropout.")
    parser.add_argument(
        "--depth",
        type=int,
        default=18,
        choices=(18, 34),
        help="ResNet depth.",
    )
    parser.add_argument(
        "--encoder1_ckpt",
        default=None,
        help="Optional pretrained encoder for the generated postoperative mask.",
    )
    parser.add_argument(
        "--encoder2_ckpt",
        default=None,
        help="Optional pretrained encoder for the preoperative volume.",
    )
    parser.add_argument(
        "--freeze_encoders",
        action="store_true",
        help=(
            "Freeze both encoders completely and train only the "
            "clinical/fusion head. Use this for the legacy released "
            "checkpoint, whose encoders were kept frozen."
        ),
    )
    parser.add_argument(
        "--unfreeze_last_block",
        dest="unfreeze_last_block",
        action="store_true",
        default=True,
        help=(
            "Freeze all encoder parameters except the last residual block "
            "(layer4). This is the paper's Stage II fine-tuning strategy."
        ),
    )
    parser.add_argument(
        "--no_unfreeze_last_block",
        dest="unfreeze_last_block",
        action="store_false",
        help="Do not unfreeze layer4; train all encoder parameters.",
    )
    parser.add_argument(
        "--init_stage2_ckpt",
        default=None,
        help=(
            "Optional full Stage II checkpoint used to initialize the model "
            "before training (for example the released multimodal weights)."
        ),
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default=None, help="Device, e.g. cuda:0 or cpu.")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision.")
    parser.add_argument(
        "--balance",
        dest="balance",
        action="store_true",
        default=True,
        help=(
            "Replicate minority classes in the training split (default: on). "
            "The paper uses a fixed, one-time oversampled training set."
        ),
    )
    parser.add_argument(
        "--no_balance",
        dest="balance",
        action="store_false",
        help="Disable training-set oversampling.",
    )
    return parser.parse_args()


def freeze_frozen_bn(model: nn.Module) -> None:
    """Keep BatchNorm layers in eval mode when their parameters are frozen."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()


def main() -> None:
    args = parse_args()
    if args.freeze_encoders and args.unfreeze_last_block:
        args.unfreeze_last_block = False
    os.makedirs(args.out_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.out_dir, "train.log"))
    save_json(vars(args), os.path.join(args.out_dir, "args.json"))
    set_seed(args.seed)

    device = get_device(args.device)
    logger.info("Device: %s", device)

    train_base = PairedCBCTDataset(
        manifest=args.manifest,
        split=args.train_split,
        image_size=args.image_size,
        balance=args.balance,
        seed=args.seed,
    )
    val_base = PairedCBCTDataset(
        manifest=args.manifest,
        split=args.val_split,
        image_size=args.image_size,
        balance=False,
        seed=args.seed,
    )

    init_state = None
    init_age_mean = None
    init_age_std = None
    if args.init_stage2_ckpt:
        init_payload = torch.load(args.init_stage2_ckpt, map_location=device)
        init_meta = init_payload if isinstance(init_payload, dict) else {}
        if "state_dict" in init_meta:
            init_state = init_meta["state_dict"]
        else:
            init_state = init_payload
        init_age_mean = init_meta.get("clinical_age_mean")
        init_age_std = init_meta.get("clinical_age_std")

    clinical_data, age_mean, age_std = load_clinical_csv(
        args.clinical_csv,
        fit_case_ids=train_base.unique_case_ids(),
        age_mean=init_age_mean,
        age_std=init_age_std,
    )
    train_dataset = MultimodalDataset(train_base, clinical_data)
    val_dataset = MultimodalDataset(val_base, clinical_data)

    missing_train = sum(
        1 for case_id in train_base.unique_case_ids()
        if case_id not in clinical_data
    )
    missing_val = sum(
        1 for case_id in val_base.unique_case_ids()
        if case_id not in clinical_data
    )
    logger.info(
        "Clinical data: %d records | age mean=%.2f std=%.2f | "
        "missing train=%d val=%d",
        len(clinical_data),
        age_mean,
        age_std,
        missing_train,
        missing_val,
    )
    logger.info("Training samples: %d %s", len(train_dataset), train_base.class_counts())
    logger.info("Validation samples: %d %s", len(val_dataset), val_base.class_counts())
    logger.info(
        "Training entries after fixed oversampling: %d (raw: %d) | "
        "balanced class counts: %s",
        len(train_dataset),
        len(train_base),
        train_base.selected_class_counts(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Frozen Stage I.
    stage1 = ARPNetStageI(
        image_size=args.image_size,
        down_scale=args.down_scale,
    ).to(device)
    payload = torch.load(args.stage1_ckpt, map_location=device)
    stage1_loss = args.stage1_loss
    if isinstance(payload, dict) and "state_dict" in payload:
        stage1_loss = payload.get("loss", stage1_loss)
        payload = payload["state_dict"]
    stage1.load_state_dict(payload)
    stage1.eval()
    for parameter in stage1.parameters():
        parameter.requires_grad = False
    logger.info("Loaded Stage I checkpoint: %s", args.stage1_ckpt)
    logger.info(
        "Stage I output activation: %s",
        "sigmoid" if stage1_loss != "mse" else "raw",
    )

    model = ARPNetStageII(
        num_classes=3,
        in_channels=1,
        depth=args.depth,
        clinical_dim=CLINICAL_DIM,
        clinical_hidden=args.clinical_hidden,
        hidden_dim=args.hidden_dim,
        fusion_dim=args.fusion_dim,
        dropout=args.dropout,
        encoder1_ckpt=args.encoder1_ckpt,
        encoder2_ckpt=args.encoder2_ckpt,
        freeze_encoders=args.freeze_encoders,
        unfreeze_last_block=args.unfreeze_last_block,
    ).to(device)

    if init_state is not None:
        model.load_state_dict(init_state, strict=True)
        logger.info("Initialized Stage II from %s", args.init_stage2_ckpt)

    if (
        (args.freeze_encoders or args.unfreeze_last_block)
        and not args.encoder1_ckpt
        and not args.encoder2_ckpt
        and not args.init_stage2_ckpt
    ):
        raise ValueError(
            "No encoder checkpoints or full Stage II initialization were "
            "provided. Provide --encoder1_ckpt/--encoder2_ckpt or "
            "--init_stage2_ckpt, or pass --no_unfreeze_last_block to train "
            "all encoder parameters from scratch."
        )

    logger.info("Trainable parameters: %s", f"{count_parameters(model):,}")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    class_weights = torch.tensor(args.class_weights, device=device)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_balanced_accuracy": [],
        "val_auroc": [],
    }
    early_stopping = EarlyStopping(
        patience=args.patience,
        mode="min",
    )

    for epoch in range(args.epochs):
        model.train()
        freeze_frozen_bn(model)
        loss_meter = AverageMeter()
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs} train",
            leave=False,
        )
        for pre, _, label, clinical, _ in progress:
            pre = pre.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            clinical = clinical.to(device, non_blocking=True)

            with torch.no_grad():
                post_pred = stage1(pre)
                if stage1_loss != "mse":
                    post_pred = torch.sigmoid(post_pred)
                post_pred = (post_pred >= 0.5).float()

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(pre, post_pred, clinical)
                if args.loss == "mse":
                    loss = multiclass_brier(logits, label)
                else:
                    loss = weighted_cross_entropy(logits, label, class_weights)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip,
                )
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), n=pre.shape[0])
            progress.set_postfix(loss=f"{loss_meter.average:.5f}")

        scheduler.step()

        model.eval()
        val_loss_meter = AverageMeter()
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for pre, _, label, clinical, _ in tqdm(
                val_loader,
                desc=f"Epoch {epoch + 1}/{args.epochs} val",
                leave=False,
            ):
                pre = pre.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                clinical = clinical.to(device, non_blocking=True)

                post_pred = stage1(pre)
                if stage1_loss != "mse":
                    post_pred = torch.sigmoid(post_pred)
                post_pred = (post_pred >= 0.5).float()

                logits = model(pre, post_pred, clinical)
                if args.loss == "mse":
                    loss = multiclass_brier(logits, label)
                else:
                    loss = weighted_cross_entropy(logits, label, class_weights)
                val_loss_meter.update(loss.item(), n=pre.shape[0])

                probabilities = torch.softmax(logits, dim=1)
                predictions = probabilities.argmax(dim=1)
                y_true.extend(label.cpu().numpy().tolist())
                y_pred.extend(predictions.cpu().numpy().tolist())
                y_prob.extend(probabilities.cpu().numpy().tolist())

        y_true_array = np.asarray(y_true)
        y_pred_array = np.asarray(y_pred)
        y_prob_array = np.asarray(y_prob)
        accuracy = float((y_true_array == y_pred_array).mean())
        balanced = float(balanced_accuracy_score(y_true_array, y_pred_array))
        metrics = classification_metrics(
            y_true_array,
            y_pred_array,
            y_prob_array,
        )

        history["train_loss"].append(loss_meter.average)
        history["val_loss"].append(val_loss_meter.average)
        history["val_accuracy"].append(accuracy)
        history["val_balanced_accuracy"].append(balanced)
        history["val_auroc"].append(metrics["auroc_macro"])

        logger.info(
            "Epoch %03d | train_loss=%.6f | val_loss=%.6f | "
            "val_acc=%.4f | val_bal_acc=%.4f | val_auroc=%.4f",
            epoch + 1,
            loss_meter.average,
            val_loss_meter.average,
            accuracy,
            balanced,
            metrics["auroc_macro"],
        )

        val_loss = val_loss_meter.average
        improved = early_stopping.step(val_loss)

        if improved:
            save_checkpoint(
                model,
                os.path.join(args.out_dir, "best.pth"),
                epoch=epoch + 1,
                val_loss=val_loss,
                val_accuracy=accuracy,
                val_balanced_accuracy=balanced,
                image_size=args.image_size,
                down_scale=args.down_scale,
                stage1_loss=stage1_loss,
                stage2_loss=args.loss,
                clinical_dim=CLINICAL_DIM,
                clinical_age_mean=age_mean,
                clinical_age_std=age_std,
            )
            logger.info(
                "Saved new best checkpoint (val_loss=%.6f, val_acc=%.4f)",
                val_loss,
                accuracy,
            )

        save_checkpoint(
            model,
            os.path.join(args.out_dir, "last.pth"),
            epoch=epoch + 1,
            val_loss=val_loss,
            val_accuracy=accuracy,
            val_balanced_accuracy=balanced,
            image_size=args.image_size,
            down_scale=args.down_scale,
            stage1_loss=stage1_loss,
            stage2_loss=args.loss,
            clinical_dim=CLINICAL_DIM,
            clinical_age_mean=age_mean,
            clinical_age_std=age_std,
        )
        save_json(history, os.path.join(args.out_dir, "history.json"))

        if early_stopping.stopped:
            logger.info(
                "Early stopping at epoch %d (no validation-loss improvement "
                "for %d epochs).",
                epoch + 1,
                args.patience,
            )
            break

    logger.info(
        "Training finished. Best validation loss: %.6f",
        early_stopping.best if early_stopping.best is not None else float("nan"),
    )


if __name__ == "__main__":
    main()
