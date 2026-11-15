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
