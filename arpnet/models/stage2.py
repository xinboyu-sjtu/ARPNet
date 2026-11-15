"""Stage II of ARPNet: need-for-ancillary-bone-augmentation (NBA) prediction.

Stage II takes the preoperative CBCT mask and the Stage I predicted post-ARP
mask.  Each volume is encoded by a 3D ResNet-18; the two 512-d feature vectors
are concatenated and passed through a two-layer fully connected classifier
that outputs three NBA classes:

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
    """Dual-encoder 3D ResNet classifier for NBA prediction.

    Parameters
    ----------
    num_classes:
        Number of NBA classes (default: 3).
    in_channels:
        Number of image channels (default: 1).
    depth:
        ResNet depth, 18 or 34.
    hidden_dim:
        Width of the fusion hidden layer.
    dropout:
        Dropout probability in the fusion head.
    encoder1_ckpt, encoder2_ckpt:
        Optional paths to pretrained ``ResNet3DEncoder`` state dicts.  The
        expected keys are ``net.*`` (the format used by the original
        ``ResNet3D0_fix.pth`` / ``ResNet3D0_no_fix.pth`` checkpoints).
    freeze_encoders:
        Freeze both encoders after loading.  Only the fusion head is trained.
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

        feature_dim = 512
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
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
        """Return NBA logits.

        Parameters
        ----------
        pre:
            Preoperative volume, ``[B, 1, D, H, W]``.
        post_pred:
            Stage I predicted post-ARP mask, ``[B, 1, D, H, W]``.
        """
        # The original ARPNet checkpoint expects encoder1 to process the
        # generated postoperative mask and encoder2 the preoperative volume.
        feat_post = self.encoder1(post_pred)
        feat_pre = self.encoder2(pre)
        fused = torch.cat([feat_post, feat_pre], dim=1)
        return self.fusion(fused)
