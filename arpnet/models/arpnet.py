"""End-to-end ARPNet: Stage I morphology prediction + Stage II NBA prediction."""

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage1 import ARPNetStageI
from .stage2 import ARPNetStageII


class ARPNet(nn.Module):
    """Two-stage ARPNet.

    Stage I is frozen; it predicts the post-ARP ridge mask from the
    preoperative volume.  Stage II classifies the need for ancillary bone
    augmentation from the preoperative volume and the predicted mask.

    Parameters
    ----------
    stage1:
        Trained :class:`ARPNetStageI`.
    stage2:
        Trained :class:`ARPNetStageII`.
    stage1_use_sigmoid:
        Set to True if Stage I was trained with Dice or BCE loss (logits).
        The MSE variant used in the paper outputs raw values in [0, 1].
    mask_threshold:
        Threshold applied to the Stage I output to obtain a binary mask.
    """

    def __init__(
        self,
        stage1: ARPNetStageI,
        stage2: ARPNetStageII,
        stage1_use_sigmoid: bool = False,
        mask_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.stage1 = stage1
        self.stage2 = stage2
        self.stage1_use_sigmoid = stage1_use_sigmoid
        self.mask_threshold = mask_threshold

        for parameter in self.stage1.parameters():
            parameter.requires_grad = False
        self.stage1.eval()

    @torch.no_grad()
    def predict_mask(self, pre: torch.Tensor) -> torch.Tensor:
        """Return a binary post-ARP mask of shape ``[B, 1, D, H, W]``."""
        out = self.stage1(pre)
        if self.stage1_use_sigmoid:
            out = torch.sigmoid(out)
        return (out >= self.mask_threshold).float()

    def forward(
        self,
        pre: torch.Tensor,
        return_mask: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Return ``(nba_logits, predicted_mask)``.

        ``nba_logits`` has shape ``[B, 3]``; ``predicted_mask`` has shape
        ``[B, 1, D, H, W]``.
        """
        mask = self.predict_mask(pre)
        logits = self.stage2(pre, mask)
        if return_mask:
            return logits, mask
        return logits

    @torch.no_grad()
    def predict_proba(self, pre: torch.Tensor) -> torch.Tensor:
        """Return NBA class probabilities of shape ``[B, 3]``."""
        logits = self.forward(pre)
        return F.softmax(logits, dim=1)
