"""Stage I of ARPNet: alveolar ridge morphology prediction.

Given a preoperative cropped CBCT mask, Stage I predicts the healed
post-alveolar-ridge-preservation (post-ARP) mask at the same location.

The network has three parallel branches operating at full, 2/3 and 1/3 scale.
Each branch is a U-Mamba encoder-decoder that returns features at the branch
resolution.  The branch outputs are fused coarse-to-fine, a global residual is
added, and a final 1x1 convolution produces the predicted volume.
"""

from typing import List, Sequence

import torch
import torch.nn as nn

from .blocks import ScaleFeatureDecoder, ScaleFeatureExtractor, UpsampleConv


class ARPNetStageI(nn.Module):
    """Three-branch scale-aware network for ridge morphology prediction.

    Parameters
    ----------
    image_size:
        Spatial size of the cubic input volume (default: 192).
    down_scale:
        Number of scale branches (default: 3, i.e. full, 2/3 and 1/3).
    in_channels:
        Number of input slices.  Defaults to ``image_size`` because the axial
        dimension is used as the channel dimension.
    features_per_stage:
        Channel widths of the four encoder stages.
    kernel_sizes, strides, n_blocks_per_stage:
        Encoder configuration shared by all branches.
    verbose:
        Print feature-map sizes and token types during construction.

    Input
    -----
    ``x`` of shape ``[B, 1, D, H, W]`` with ``D = H = W = image_size``.
    A 4D tensor ``[B, D, H, W]`` is also accepted.

    Output
    ------
    ``[B, 1, D, H, W]`` raw mask logits/values.  For the MSE variant used in
    the paper, threshold the raw output at 0.5.  For Dice/BCE variants, apply
    a sigmoid before thresholding.
    """

    def __init__(
        self,
        image_size: int = 192,
        down_scale: int = 3,
        in_channels: int = None,
        features_per_stage: Sequence[int] = (64, 128, 192, 256),
        kernel_sizes: Sequence[int] = (3, 5, 7, 3),
        strides: Sequence[int] = (2, 2, 2, 2),
        n_blocks_per_stage: Sequence[int] = (2, 2, 2, 2),
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.down_scale = int(down_scale)
        self.in_channels = (
            self.image_size if in_channels is None else int(in_channels)
        )

        n_stages = len(features_per_stage)
        stride_product = 2 ** len(strides)
        if self.image_size % (self.down_scale * stride_product) != 0:
            raise ValueError(
                "image_size must be divisible by down_scale * "
                f"2**n_stages ({self.down_scale * stride_product}) so that "
                "every scale branch has consistent feature-map sizes; "
                f"got image_size={self.image_size}."
            )
        branches = []
        for branch_idx in range(self.down_scale):
            scale_num = self.down_scale - branch_idx
            branch_channels = self.in_channels * scale_num // self.down_scale
            branch_size = self.image_size * scale_num // self.down_scale

            encoder = ScaleFeatureExtractor(
                input_size=(branch_size, branch_size),
                input_channels=branch_channels,
                n_stages=n_stages,
                features_per_stage=list(features_per_stage),
                conv_op=nn.Conv2d,
                kernel_sizes=list(kernel_sizes),
                strides=list(strides),
                n_blocks_per_stage=list(n_blocks_per_stage),
                conv_bias=True,
                norm_op=nn.BatchNorm2d,
                norm_op_kwargs={"eps": 1e-5, "momentum": 0.1},
                nonlin=nn.LeakyReLU,
                nonlin_kwargs={"inplace": True},
                return_skips=True,
                verbose=verbose,
            )
            decoder = ScaleFeatureDecoder(
                encoder=encoder,
                out_channels=branch_channels,
                n_conv_per_stage=2,
            )
            upsampler = UpsampleConv(
                in_channels=branch_channels,
                out_channels=branch_channels,
                stride=2,
            )
            branches.append(nn.ModuleList([encoder, decoder, upsampler]))

        self.downscale_branches = nn.ModuleList(branches)

        # 1x1 convolutions applied after each coarse-to-fine fusion step.
        # Index 0 handles the full-scale branch, index 1 the 2/3 branch, etc.
        self.scale_feature_fusion_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.in_channels
                        * (self.down_scale - i)
                        // self.down_scale,
                        out_channels=self.in_channels
                        * (self.down_scale - i)
                        // self.down_scale,
                        kernel_size=1,
                    )
                )
                for i in range(self.down_scale - 1)
            ]
        )

        self.refiner = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=1,
        )

    def _crop_scale(self, x: torch.Tensor, scale_num: int) -> torch.Tensor:
        """Return the centre crop of ``x`` for a branch at ``scale_num``."""
        c0 = int(
            self.in_channels // 2
            - self.in_channels * scale_num // (2 * self.down_scale)
        )
        c1 = int(
            self.in_channels // 2
            + self.in_channels * scale_num // (2 * self.down_scale)
        )
        s0 = int(
            self.image_size // 2
            - self.image_size * scale_num // (2 * self.down_scale)
        )
        s1 = int(
            self.image_size // 2
            + self.image_size * scale_num // (2 * self.down_scale)
        )
        return x[:, c0:c1, s0:s1, s0:s1]

    def scale_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run all scale branches and fuse them coarse-to-fine."""
        outputs: List[torch.Tensor] = []
        for branch_idx in range(self.down_scale):
            x_i = self._crop_scale(x, self.down_scale - branch_idx)
            encoder, decoder, upsampler = self.downscale_branches[branch_idx]
            outputs.append(upsampler(decoder(encoder(x_i))))

        # Fuse the smallest branch into the next larger branch, and so on.
        for step in range(1, self.down_scale):
            target_idx = self.down_scale - step - 1
            start = int(
                (self.image_size * (step + 1)) // (2 * self.down_scale)
                - (self.image_size * step) // (2 * self.down_scale)
            )
            end = int(
                (self.image_size * (step + 1)) // (2 * self.down_scale)
                + (self.image_size * step) // (2 * self.down_scale)
            )
            start_c = self.in_channels // (2 * self.down_scale)
            end_c = (
                self.in_channels * (step + 1) // self.down_scale
                - self.in_channels // (2 * self.down_scale)
            )

            target = outputs[target_idx]
            target[:, start_c:end_c, start:end, start:end] = (
                target[:, start_c:end_c, start:end, start:end]
                + outputs[self.down_scale - step]
            )
            outputs[target_idx] = self.scale_feature_fusion_layers[target_idx](
                target
            )

        return outputs[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            if x.shape[1] != 1:
                raise ValueError(
                    "ARPNetStageI expects a single input channel, "
                    f"got {x.shape[1]}."
                )
            x = x[:, 0]
        if x.ndim != 4:
            raise ValueError(
                "ARPNetStageI expects [B, 1, D, H, W] or [B, D, H, W], "
                f"got shape {tuple(x.shape)}."
            )
        if x.shape[1:] != (self.in_channels, self.image_size, self.image_size):
            raise ValueError(
                "Input shape does not match the configured Stage I size: "
                f"expected [B, {self.in_channels}, {self.image_size}, "
                f"{self.image_size}], got {tuple(x.shape)}."
            )

        out = self.scale_forward(x) + x
        out = self.refiner(out)
        return out.unsqueeze(1)
