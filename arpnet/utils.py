"""Shared utilities: seeding, logging, checkpointing and device selection."""

import json
import logging
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set Python, NumPy and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(requested: Optional[str] = None) -> torch.device:
    """Return a CUDA device when available, otherwise CPU."""
    if requested:
        return torch.device(requested)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_logger(log_path: str, name: str = "arpnet") -> logging.Logger:
    """Create a logger that writes to both a file and the console."""
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def save_json(data: Dict[str, Any], path: str) -> None:
    """Write a dictionary as pretty-printed JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    device: torch.device,
    strict: bool = True,
) -> None:
    """Load a model state dict from ``path``.

    Both a raw state dict and a dictionary containing a ``state_dict`` key are
    accepted.
    """
    state_dict = torch.load(path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=strict)


def save_checkpoint(
    model: torch.nn.Module,
    path: str,
    **extra: Any,
) -> None:
    """Save a model state dict together with optional metadata."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {"state_dict": model.state_dict(), **extra}
    torch.save(payload, path)


def count_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    """Track the running mean of a scalar."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else 0.0


def validate_encoder_initialization(
    encoder1_ckpt: Optional[str],
    encoder2_ckpt: Optional[str],
    init_stage2_ckpt: Optional[str],
    freeze_encoders: bool,
    unfreeze_last_block: bool,
) -> None:
    """Require both encoders when a frozen/fine-tuning strategy is used.

    A full Stage II checkpoint initializes both encoders and is therefore a
    valid alternative to the two standalone encoder checkpoints.
    """
    if not (freeze_encoders or unfreeze_last_block):
        return
    if init_stage2_ckpt:
        return

    missing = []
    if not encoder1_ckpt:
        missing.append("--encoder1_ckpt")
    if not encoder2_ckpt:
        missing.append("--encoder2_ckpt")
    if missing:
        raise ValueError(
            "Both encoder checkpoints are required when freeze_encoders or "
            "unfreeze_last_block is enabled, because otherwise one encoder "
            "would remain randomly initialized and frozen. Missing: "
            + ", ".join(missing)
            + ". Provide both checkpoints, provide --init_stage2_ckpt, "
            "or pass --no_unfreeze_last_block to train all encoder "
            "parameters from scratch."
        )


class EarlyStopping:
    """Early stopping on a validation metric.

    Parameters
    ----------
    patience:
        Number of epochs without improvement before stopping.
    min_delta:
        Minimum change to qualify as an improvement.
    mode:
        ``"min"`` for losses, ``"max"`` for scores.
    """

    def __init__(
        self,
        patience: int = 50,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'.")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best: Optional[float] = None
        self.counter = 0
        self.stopped = False

    def step(self, value: float) -> bool:
        """Update the state and return True if the metric improved."""
        value = float(value)
        if not np.isfinite(value):
            self.counter += 1
            self.stopped = self.counter >= self.patience
            return False

        if self.best is None:
            self.best = value
            self.counter = 0
            return True

        if self.mode == "min":
            improved = value < self.best - self.min_delta
        else:
            improved = value > self.best + self.min_delta

        if improved:
            self.best = value
            self.counter = 0
            return True

        self.counter += 1
        self.stopped = self.counter >= self.patience
        return False
