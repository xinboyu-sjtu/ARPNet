"""Building blocks for the Stage I morphology-prediction network.

The Stage I network is a three-branch, scale-aware encoder-decoder.  Each
branch receives a centre crop of the input volume at a different scale (full,
2/3 and 1/3) and is encoded by a U-Mamba-style backbone: residual convolutional
stages interleaved with Mamba state-space layers.  The branch outputs are then
fused coarse-to-fine by 1x1 convolutions.

Volume convention
-----------------
A 3D volume ``[D, H, W]`` is processed as a 2D multi-channel image
``[C=D, H, W]``.  The axial dimension therefore becomes the channel dimension,
which keeps memory usage manageable while still producing a full 3D mask.
"""

from typing import List, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn.modules.conv import _ConvNd

from dynamic_network_architectures.building_blocks.helper import (
    maybe_convert_scalar_to_list,
)
from dynamic_network_architectures.building_blocks.residual import BasicBlockD
from mamba_ssm import Mamba


class MambaLayer(nn.Module):
    """LayerNorm followed by a Mamba selective state-space layer.

    Parameters
    ----------
    dim:
        Feature dimension.
    d_state, d_conv, expand:
        Mamba hyper-parameters.
    channel_token:
        If True, the channel dimension is treated as the token dimension.
        Otherwise the spatial positions are flattened into tokens.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        channel_token: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.channel_token = channel_token

    def forward_patch_token(self, x: torch.Tensor) -> torch.Tensor:
        batch, dim = x.shape[:2]
        assert dim == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]

        x_flat = x.reshape(batch, dim, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        return x_mamba.transpose(-1, -2).reshape(batch, dim, *img_dims)

    def forward_channel_token(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_tokens = x.shape[:2]
        dim = x.shape[2:].numel()
        assert dim == self.dim, f"Expected dim={self.dim}, got {dim}"
        img_dims = x.shape[2:]

        x_flat = x.flatten(2)
        assert x_flat.shape[2] == dim
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        return x_mamba.reshape(batch, n_tokens, *img_dims)

    @autocast(enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mamba is numerically stable in float32; autocast is disabled here.
        if x.dtype == torch.float16:
            x = x.float()
        if self.channel_token:
            return self.forward_channel_token(x)
        return self.forward_patch_token(x)


class BasicResBlock(nn.Module):
    """Two 3x3 convolutions with BatchNorm and a residual connection."""

    def __init__(
        self,
        conv_op,
        input_channels: int,
        output_channels: int,
        norm_op,
        norm_op_kwargs,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        use_1x1conv: bool = False,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs=None,
    ) -> None:
        super().__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {"inplace": True}

        self.conv1 = conv_op(
            input_channels,
            output_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)

        self.conv2 = conv_op(
            output_channels,
            output_channels,
            kernel_size,
            padding=padding,
        )
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)

        if use_1x1conv:
            self.conv3 = conv_op(
                input_channels,
                output_channels,
                kernel_size=1,
                stride=stride,
            )
        else:
            self.conv3 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3 is not None:
            x = self.conv3(x)
        y = y + x
        return self.act2(y)


class ScaleFeatureExtractor(nn.Module):
    """U-Mamba-style encoder used by one scale branch.

    The encoder starts with a stem, then applies ``n_stages`` residual stages.
    A Mamba layer is inserted after every other stage so that the last stage
    always contains one.  Skip tensors from all stages are returned for the
    decoder.
    """

    def __init__(
        self,
        input_size: Tuple[int, ...],
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = True,
        norm_op=None,
        norm_op_kwargs=None,
        nonlin=None,
        nonlin_kwargs=None,
        return_skips: bool = True,
        verbose: bool = False,
    ) -> None:
        super().__init__()

        kernel_sizes = self._to_list(kernel_sizes, n_stages)
        features_per_stage = self._to_list(features_per_stage, n_stages)
        n_blocks_per_stage = self._to_list(n_blocks_per_stage, n_stages)
        strides = self._to_list(strides, n_stages)

        assert len(kernel_sizes) == n_stages
        assert len(features_per_stage) == n_stages
        assert len(n_blocks_per_stage) == n_stages
        assert len(strides) == n_stages

        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        # Decide whether each stage uses channel tokens or patch tokens.
        # Channel tokens are used once the spatial feature map is small enough.
        do_channel_token = [False] * n_stages
        feature_map_sizes = []
        feature_map_size = list(input_size)
        for stage in range(n_stages):
            feature_map_size = [
                i // j for i, j in zip(feature_map_size, strides)
            ]
            feature_map_sizes.append(feature_map_size)
            if int(np.prod(feature_map_size)) <= features_per_stage[stage]:
                do_channel_token[stage] = True

        if verbose:
            print(f"feature_map_sizes: {feature_map_sizes}")
            print(f"do_channel_token: {do_channel_token}")

        self.conv_pad_sizes = [kernel_size // 2 for kernel_size in kernel_sizes]
        self.return_skips = return_skips

        # Stem
        stem_channels = features_per_stage[0]
        self.stem = nn.Sequential(
            BasicResBlock(
                conv_op=conv_op,
                input_channels=input_channels,
                output_channels=stem_channels,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                kernel_size=kernel_sizes[0],
                padding=self.conv_pad_sizes[0],
                stride=1,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                use_1x1conv=True,
            ),
            *[
                BasicBlockD(
                    conv_op=conv_op,
                    input_channels=stem_channels,
                    output_channels=stem_channels,
                    kernel_size=kernel_sizes[0],
                    stride=1,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                )
                for _ in range(n_blocks_per_stage[0] - 1)
            ],
        )

        # Encoder stages
        stages = []
        mamba_layers = []
        current_channels = stem_channels
        for stage in range(n_stages):
            block = nn.Sequential(
                BasicResBlock(
                    conv_op=conv_op,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    input_channels=current_channels,
                    output_channels=features_per_stage[stage],
                    kernel_size=kernel_sizes[stage],
                    padding=self.conv_pad_sizes[stage],
                    stride=strides[stage],
                    use_1x1conv=True,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                ),
                *[
                    BasicBlockD(
                        conv_op=conv_op,
                        input_channels=features_per_stage[stage],
                        output_channels=features_per_stage[stage],
                        kernel_size=kernel_sizes[stage],
                        stride=1,
                        conv_bias=conv_bias,
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs,
                    )
                    for _ in range(n_blocks_per_stage[stage] - 1)
                ],
            )

            # Insert Mamba in every other stage, always including the last one.
            if bool(stage % 2) ^ bool(n_stages % 2):
                mamba_layers.append(
                    MambaLayer(
                        dim=(
                            int(np.prod(feature_map_sizes[stage]))
                            if do_channel_token[stage]
                            else features_per_stage[stage]
                        ),
                        channel_token=do_channel_token[stage],
                    )
                )
            else:
                mamba_layers.append(nn.Identity())

            stages.append(block)
            current_channels = features_per_stage[stage]

        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.stages = nn.ModuleList(stages)
        self.output_channels = features_per_stage
        self.strides = [
            maybe_convert_scalar_to_list(conv_op, stride) for stride in strides
        ]

        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    @staticmethod
    def _to_list(value, length: int) -> List:
        if isinstance(value, int):
            return [value] * length
        return list(value)

    def forward(self, x: torch.Tensor):
        if self.stem is not None:
            x = self.stem(x)

        skips = []
        for stage in range(len(self.stages)):
            x = self.stages[stage](x)
            x = self.mamba_layers[stage](x)
            skips.append(x)

        return skips if self.return_skips else skips[-1]


class UpsampleLayer(nn.Module):
    """Nearest-neighbour upsampling followed by a 1x1 convolution."""

    def __init__(
        self,
        conv_op,
        input_channels: int,
        output_channels: int,
        scale_factor,
        mode: str = "nearest",
    ) -> None:
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1)
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return self.conv(x)


class ScaleFeatureDecoder(nn.Module):
    """Decoder that fuses encoder skip connections at every resolution."""

    def __init__(
        self,
        encoder: ScaleFeatureExtractor,
        out_channels: int,
        n_conv_per_stage: Union[int, Tuple[int, ...], List[int]] = 2,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        n_stages_encoder = len(encoder.output_channels)

        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1

        stages = []
        upsample_layers = []
        seg_layers = []

        for stage in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-stage]
            input_features_skip = encoder.output_channels[-(stage + 1)]
            stride_for_upsampling = encoder.strides[-stage]

            upsample_layers.append(
                UpsampleLayer(
                    conv_op=encoder.conv_op,
                    input_channels=input_features_below,
                    output_channels=input_features_skip,
                    scale_factor=stride_for_upsampling,
                    mode="nearest",
                )
            )

            stages.append(
                nn.Sequential(
                    BasicResBlock(
                        conv_op=encoder.conv_op,
                        norm_op=encoder.norm_op,
                        norm_op_kwargs=encoder.norm_op_kwargs,
                        nonlin=encoder.nonlin,
                        nonlin_kwargs=encoder.nonlin_kwargs,
                        input_channels=(
                            2 * input_features_skip
                            if stage < n_stages_encoder - 1
                            else input_features_skip
                        ),
                        output_channels=input_features_skip,
                        kernel_size=encoder.kernel_sizes[-(stage + 1)],
                        padding=encoder.conv_pad_sizes[-(stage + 1)],
                        stride=1,
                        use_1x1conv=True,
                    ),
                    *[
                        BasicBlockD(
                            conv_op=encoder.conv_op,
                            input_channels=input_features_skip,
                            output_channels=input_features_skip,
                            kernel_size=encoder.kernel_sizes[-(stage + 1)],
                            stride=1,
                            conv_bias=encoder.conv_bias,
                            norm_op=encoder.norm_op,
                            norm_op_kwargs=encoder.norm_op_kwargs,
                            nonlin=encoder.nonlin,
                            nonlin_kwargs=encoder.nonlin_kwargs,
                        )
                        for _ in range(n_conv_per_stage[stage - 1] - 1)
                    ],
                )
            )
            seg_layers.append(
                encoder.conv_op(
                    input_features_skip,
                    out_channels,
                    1,
                    1,
                    0,
                    bias=True,
                )
            )

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips: List[torch.Tensor]) -> torch.Tensor:
        low_res = skips[-1]
        out = None
        for stage in range(len(self.stages)):
            x = self.upsample_layers[stage](low_res)
            if stage < len(self.stages) - 1:
                x = torch.cat((x, skips[-(stage + 2)]), dim=1)
            x = self.stages[stage](x)
            if stage == len(self.stages) - 1:
                out = self.seg_layers[-1](x)
            low_res = x
        return out


class UpsampleConv(nn.Module):
    """Transposed convolution that doubles the spatial resolution."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=stride,
            padding=1,
            output_padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_transpose(x)
