#!/usr/bin/env python
"""Train Stage I of ARPNet (alveolar ridge morphology prediction).

Example
-------
python scripts/train_stage1.py \
    --manifest examples/manifest_example.csv \
    --out_dir runs/stage1 \
    --epochs 200 \
    --loss mse

The paper uses MSE.  ``--loss dice`` and ``--loss bce`` are provided for the
loss-function comparison reported in the revision.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from arpnet.data.dataset import PairedCBCTDataset
from arpnet.losses import bce_loss, dice_loss, mse_loss
from arpnet.metrics import dice_score_tensor
from arpnet.models.stage1 import ARPNetStageI
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
        description="Train ARPNet Stage I (ridge morphology prediction)."
    )
    parser.add_argument("--manifest", required=True, help="CSV manifest.")
    parser.add_argument("--out_dir", default="runs/stage1", help="Output directory.")
    parser.add_argument("--train_split", default="train", help="Training split name.")
    parser.add_argument("--val_split", default="val", help="Validation split name.")
    parser.add_argument("--image_size", type=int, default=192, help="Cubic volume size.")
    parser.add_argument("--down_scale", type=int, default=3, help="Number of scale branches.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs.")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument(
        "--loss",
        choices=("mse", "dice", "bce"),
        default="mse",
        help="Stage I loss. The paper uses MSE.",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default=None, help="Device, e.g. cuda:0 or cpu.")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision.")
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Replicate minority classes in the training split.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Optional Stage I checkpoint to resume from.",
    )
    return parser.parse_args()


def build_criterion(name: str):
    if name == "mse":
        return mse_loss
    if name == "dice":
        return dice_loss
    if name == "bce":
        return bce_loss
    raise ValueError(f"Unsupported loss: {name}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.out_dir, "train.log"))
    save_json(vars(args), os.path.join(args.out_dir, "args.json"))
    set_seed(args.seed)

    device = get_device(args.device)
    logger.info("Device: %s", device)

    train_dataset = PairedCBCTDataset(
        manifest=args.manifest,
        split=args.train_split,
        image_size=args.image_size,
        balance=args.balance,
        seed=args.seed,
    )
    val_dataset = PairedCBCTDataset(
        manifest=args.manifest,
        split=args.val_split,
        image_size=args.image_size,
        balance=False,
        seed=args.seed,
    )
    logger.info("Training samples: %d %s", len(train_dataset), train_dataset.class_counts())
    logger.info("Validation samples: %d %s", len(val_dataset), val_dataset.class_counts())

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

    model = ARPNetStageI(
        image_size=args.image_size,
        down_scale=args.down_scale,
    ).to(device)
    logger.info("Trainable parameters: %s", f"{count_parameters(model):,}")

    if args.resume:
        state_dict = torch.load(args.resume, map_location=device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)
        logger.info("Resumed from %s", args.resume)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )
    criterion = build_criterion(args.loss)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    history = {"train_loss": [], "val_dice": [], "val_loss": []}
    early_stopping = EarlyStopping(
        patience=args.patience,
        mode="min",
    )

    for epoch in range(args.epochs):
        model.train()
        loss_meter = AverageMeter()
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs} train",
            leave=False,
        )
        for pre, post, _, _ in progress:
            pre = pre.to(device, non_blocking=True)
            post = post.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                prediction = model(pre)
                loss = criterion(prediction, post)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                )
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), n=pre.shape[0])
            progress.set_postfix(loss=f"{loss_meter.average:.5f}")

        scheduler.step()

        model.eval()
        val_dices = []
        val_loss_meter = AverageMeter()
        with torch.no_grad():
            for pre, post, _, _ in tqdm(
                val_loader,
                desc=f"Epoch {epoch + 1}/{args.epochs} val",
                leave=False,
            ):
                pre = pre.to(device, non_blocking=True)
                post = post.to(device, non_blocking=True)
                prediction = model(pre)
                val_loss_meter.update(
                    criterion(prediction, post).item(),
                    n=pre.shape[0],
                )
                val_dices.append(
                    dice_score_tensor(
                        prediction,
                        post,
                        from_logits=(args.loss != "mse"),
                    )
                )

        mean_dice = float(np.mean(val_dices)) if val_dices else float("nan")
        history["train_loss"].append(loss_meter.average)
        history["val_loss"].append(val_loss_meter.average)
        history["val_dice"].append(mean_dice)

        logger.info(
            "Epoch %03d | train_loss=%.6f | val_loss=%.6f | val_dice=%.4f",
            epoch + 1,
            loss_meter.average,
            val_loss_meter.average,
            mean_dice,
        )

        val_loss = val_loss_meter.average
        improved = early_stopping.step(val_loss)

        if improved:
            save_checkpoint(
                model,
                os.path.join(args.out_dir, "best.pth"),
                epoch=epoch + 1,
                val_loss=val_loss,
                val_dice=mean_dice,
                image_size=args.image_size,
                down_scale=args.down_scale,
                loss=args.loss,
            )
            logger.info(
                "Saved new best checkpoint (val_loss=%.6f, val_dice=%.4f)",
                val_loss,
                mean_dice,
            )

        save_checkpoint(
            model,
            os.path.join(args.out_dir, "last.pth"),
            epoch=epoch + 1,
            val_loss=val_loss,
            val_dice=mean_dice,
            image_size=args.image_size,
            down_scale=args.down_scale,
            loss=args.loss,
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
