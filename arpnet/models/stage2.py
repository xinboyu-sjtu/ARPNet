"""Stage II of ARPNet: multimodal NBA prediction.

Stage II combines three sources of information:

1. the preoperative CBCT mask,
2. the Stage I predicted post-ARP mask,
3. an 8-dimensional clinical vector (tooth type, jaw, gender, age).

Each volume is encoded by a 3D ResNet-18.  The two 512-d image features are
concatenated with a 64-d clinical embedding, and the resulting 1088-d vector is
passed through a two-layer fully connected classifier:

    1088 -> 1024 -> 512 -> 3

The three NBA classes are:

    class 0: implant placement without ancillary bone augmentation
    class 1: implant placement with simultaneous bone augmentation / OSFE
    class 2: implant placement after complex bone augmentation (staged GBR/LSFE)
"""

import os
from typing import Optional

import torch
import torch.nn as nn
from monai.networks.nets import resnet


class ResNet3DEncoder(nn.Module):
    """3D ResNet-18 feature extractor.

    The classification head (``net.fc``) is kept for checkpoint compatibility
    with the original implementation, but it is not used: features are taken
    after global average pooling.
    """

    def __init__(
        self,
        in_channels: int = 1,
        depth: int = 18,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        if depth == 18:
            self.net = resnet.resnet18(
                spatial_dims=3,
                n_input_channels=in_channels,
                num_classes=num_classes,
            )
        elif depth == 34:
            self.net = resnet.resnet34(
                spatial_dims=3,
                n_input_channels=in_channels,
                num_classes=num_classes,
            )
        else:
            raise ValueError("Only ResNet-18 and ResNet-34 are supported.")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.act(x)
        if not self.net.no_max_pool:
            x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = self.net.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


class ARPNetStageII(nn.Module):
    """Multimodal dual-encoder 3D ResNet classifier for NBA prediction.

    Parameters
    ----------
    num_classes:
        Number of NBA classes (default: 3).
    in_channels:
        Number of image channels (default: 1).
    depth:
        ResNet depth, 18 or 34.
    clinical_dim:
        Dimension of the clinical vector (default: 8).
    clinical_hidden:
        Width of the clinical embedding (default: 64).
    hidden_dim:
        Width of the first fusion layer (default: 1024).
    fusion_dim:
        Width of the second fusion layer (default: 512).
    dropout:
        Dropout probability in the fusion head (default: 0.3).
    encoder1_ckpt, encoder2_ckpt:
        Optional pretrained ``ResNet3DEncoder`` state dicts.  The expected keys
        are ``net.*``.  ``encoder1`` processes the generated postoperative
        mask; ``encoder2`` processes the preoperative volume.
    freeze_encoders:
        Freeze both encoders completely.  This matches the legacy image-only
        and multimodal checkpoints released before the paper-aligned training
        configuration.
    unfreeze_last_block:
        Freeze all encoder parameters except the last residual block
        (``layer4``) of each encoder.  This is the paper's Stage II
        fine-tuning strategy when pretrained encoder weights are available.
        ``freeze_encoders`` and ``unfreeze_last_block`` are mutually exclusive.
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 1,
        depth: int = 18,
        clinical_dim: int = 8,
        clinical_hidden: int = 64,
        hidden_dim: int = 1024,
        fusion_dim: int = 512,
        dropout: float = 0.3,
        encoder1_ckpt: Optional[str] = None,
        encoder2_ckpt: Optional[str] = None,
        freeze_encoders: bool = False,
        unfreeze_last_block: bool = False,
    ) -> None:
        super().__init__()
        if freeze_encoders and unfreeze_last_block:
            raise ValueError(
                "freeze_encoders and unfreeze_last_block are mutually exclusive."
            )
        self.encoder1 = ResNet3DEncoder(
            in_channels=in_channels,
            depth=depth,
            num_classes=num_classes,
        )
        self.encoder2 = ResNet3DEncoder(
            in_channels=in_channels,
            depth=depth,
            num_classes=num_classes,
        )

        self._load_encoder(self.encoder1, encoder1_ckpt)
        self._load_encoder(self.encoder2, encoder2_ckpt)

        self.clinical_proj = nn.Sequential(
            nn.LayerNorm(clinical_dim),
            nn.Linear(clinical_dim, clinical_hidden),
            nn.ReLU(inplace=True),
        )

        fusion_in = 512 * 2 + clinical_hidden
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

        if freeze_encoders:
            for parameter in self.encoder1.parameters():
                parameter.requires_grad = False
            for parameter in self.encoder2.parameters():
                parameter.requires_grad = False
        elif unfreeze_last_block:
            for parameter in self.encoder1.parameters():
                parameter.requires_grad = False
            for parameter in self.encoder2.parameters():
                parameter.requires_grad = False
            for parameter in self.encoder1.net.layer4.parameters():
                parameter.requires_grad = True
            for parameter in self.encoder2.net.layer4.parameters():
                parameter.requires_grad = True

    @staticmethod
    def _load_encoder(encoder: ResNet3DEncoder, path: Optional[str]) -> None:
        if path is None:
            return
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Encoder checkpoint not found: {path}")
        state_dict = torch.load(path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        encoder.load_state_dict(state_dict, strict=True)

    def forward(
        self,
        pre: torch.Tensor,
        post_pred: torch.Tensor,
        clinical: torch.Tensor,
    ) -> torch.Tensor:
        """Return NBA logits.

        Parameters
        ----------
        pre:
            Preoperative volume, ``[B, 1, D, H, W]``.
        post_pred:
            Stage I predicted post-ARP mask, ``[B, 1, D, H, W]``.
        clinical:
            Clinical vector, ``[B, clinical_dim]``.
        """
        # The original ARPNet checkpoint expects encoder1 to process the
        # generated postoperative mask and encoder2 the preoperative volume.
        feat_post = self.encoder1(post_pred)
        feat_pre = self.encoder2(pre)
        feat_clinical = self.clinical_proj(clinical)
        fused = torch.cat([feat_post, feat_pre, feat_clinical], dim=1)
        return self.fusion(fused)


class ARPNetStageIIImageOnly(nn.Module):
    """Image-only dual-encoder control model for NBA prediction.

    This is the earlier image-only Stage II architecture: two 3D ResNet-18
    encoders (512 features each) and a ``1024 -> 256 -> 3`` classifier with
    dropout 0.1.  It is kept as an ablation/control model and for compatibility
    with the legacy image-only checkpoint.
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 1,
        depth: int = 18,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        encoder1_ckpt: Optional[str] = None,
        encoder2_ckpt: Optional[str] = None,
        freeze_encoders: bool = False,
    ) -> None:
        super().__init__()
        self.encoder1 = ResNet3DEncoder(
            in_channels=in_channels,
            depth=depth,
            num_classes=num_classes,
        )
        self.encoder2 = ResNet3DEncoder(
            in_channels=in_channels,
            depth=depth,
            num_classes=num_classes,
        )

        self._load_encoder(self.encoder1, encoder1_ckpt)
        self._load_encoder(self.encoder2, encoder2_ckpt)

        self.fusion = nn.Sequential(
            nn.Linear(512 * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        if freeze_encoders:
            for parameter in self.encoder1.parameters():
                parameter.requires_grad = False
            for parameter in self.encoder2.parameters():
                parameter.requires_grad = False

    @staticmethod
    def _load_encoder(encoder: ResNet3DEncoder, path: Optional[str]) -> None:
        if path is None:
            return
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Encoder checkpoint not found: {path}")
        state_dict = torch.load(path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        encoder.load_state_dict(state_dict, strict=True)

    def forward(
        self,
        pre: torch.Tensor,
        post_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Return NBA logits without clinical variables."""
        feat_post = self.encoder1(post_pred)
        feat_pre = self.encoder2(pre)
        fused = torch.cat([feat_post, feat_pre], dim=1)
        return self.fusion(fused)
