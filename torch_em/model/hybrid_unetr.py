"""UNETR with hybrid 2D/3D decoders that read multi-scale encoder features.

The decoders read the encoder features at stride 4, 8 and 16 (e.g. of the SAM2 Hiera encoder) and the preprocessed
image, from which an image stem computes the skips at stride 2 and 1. They follow the hybrid 2D/3D recipe:

- All spatial operations are 2D and run on the z-slices folded into the batch: (B, C, Z, H, W) becomes
  (B * Z, C, H, W). The normalization is a GroupNorm on the folded tensor, so its statistics are per slice.
- The context along z comes only from residual branches that are zero-initialized and skipped for a single slice:
  `ZResidual` (convolution along z) and `ZAttention` (attention along z).

The same weights therefore decode single images (Z = 1) and volumes (Z > 1). At initialization a volume decodes
exactly to the stack of its decoded slices.

The blocks and decoders take the features with z folded into the batch, (B * Z, C, H, W). The ones with
context along z also take the number of slices `n_z`.

The decoders:
- `HieraUDecoder` ("hiera_u"): a plain skip U-decoder.
- `C3F2Decoder` ("c3f2"): a coarse-3D / fine-2D fusion head. ConvNeXt blocks with attention along z at stride 16,
  with convolutions along z at strides 8 and 4, and a 2D image-guided upsampling head.
- `CAGD3DDecoder` ("cagd_3d"): the decoder of MedCAGD (Wazir et al., arXiv 2607.00409,
  https://github.com/saadwazir/MedCAGD) adapted to the hybrid recipe: gated skip fusion, context aggregators,
  refinement blocks with channel recalibration and deep supervision.

`HybridUNETR` combines an encoder with one of the decoders.
"""

import math
from collections.abc import Mapping
from os import PathLike
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unetr import UNETRBase


def fold_z(x: torch.Tensor) -> torch.Tensor:
    """Fold z into the batch: (B, C, Z, H, W) to (B * Z, C, H, W).

    Args:
        x: The input tensor.

    Returns:
        The folded tensor.
    """
    b, c, z, h, w = x.shape
    return x.transpose(1, 2).reshape(b * z, c, h, w)


def unfold_z(x: torch.Tensor, n_z: int) -> torch.Tensor:
    """Unfold z from the batch: (B * Z, C, H, W) to (B, C, Z, H, W).

    Args:
        x: The folded tensor.
        n_z: The number of slices Z.

    Returns:
        The unfolded tensor.
    """
    bz, c, h, w = x.shape
    return x.reshape(bz // n_z, n_z, c, h, w).transpose(1, 2)


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(8, channels), channels)


def _resize_to(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == reference.shape[-2:]:
        return x
    return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)


def _resize_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    return torch.cat([_resize_to(x, skip), skip], dim=1)


#
# BLOCKS
#


class ZResidual(nn.Module):
    """A residual branch along z: x + pointwise(GELU(depthwise_z(x))).

    The pointwise convolution is zero-initialized, so the branch starts as the identity.
    It is skipped for a single slice, so 2D inputs never see it.

    Args:
        channels: The number of channels.
        kernel_size: The kernel size along z.
    """
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.depthwise = nn.Conv3d(
            channels, channels, kernel_size=(kernel_size, 1, 1), padding=(kernel_size // 2, 0, 0), groups=channels,
        )
        self.act = nn.GELU()
        self.pointwise = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, x: torch.Tensor, n_z: int) -> torch.Tensor:
        if n_z == 1:
            return x
        volume = unfold_z(x, n_z)
        return x + fold_z(self.pointwise(self.act(self.depthwise(volume))))


class ZAttention(nn.Module):
    """Residual self-attention along z at every xy position, with a learned relative position bias.

    The output projection is zero-initialized, so the block starts as the identity.
    It is skipped for a single slice.

    Args:
        channels: The number of channels.
        n_heads: The number of attention heads. By default, it uses one head per 64 channels.
        max_distance: The largest z distance with its own position bias. Larger distances share the last one.
    """
    def __init__(self, channels: int, n_heads: Optional[int] = None, max_distance: int = 16):
        super().__init__()
        self.n_heads = n_heads or max(1, channels // 64)
        if channels % self.n_heads != 0:
            raise ValueError(f"{channels} channels cannot be split into {self.n_heads} heads.")
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.max_distance = max_distance
        self.position_bias = nn.Parameter(torch.zeros(self.n_heads, 2 * max_distance + 1))

    def forward(self, x: torch.Tensor, n_z: int) -> torch.Tensor:
        if n_z == 1:
            return x
        bz, c, h, w = x.shape
        b = bz // n_z
        tokens = x.reshape(b, n_z, c, h, w).permute(0, 3, 4, 1, 2).reshape(b * h * w, n_z, c)
        qkv = self.qkv(self.norm(tokens)).reshape(-1, n_z, 3, self.n_heads, c // self.n_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        index = torch.arange(n_z, device=x.device)
        distance = (index[None] - index[:, None]).clamp(-self.max_distance, self.max_distance) + self.max_distance
        bias = self.position_bias[:, distance].to(q.dtype)[None]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        out = self.proj(out.transpose(1, 2).reshape(-1, n_z, c))
        out = out.reshape(b, h, w, n_z, c).permute(0, 3, 4, 1, 2).reshape(bz, c, h, w)
        return x + out


class HybridBlock(nn.Module):
    """A residual 2D block with per-slice GroupNorm and an optional zero-initialized z branch.

    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        with_z: Whether to add the z branch.
    """
    def __init__(self, in_channels: int, out_channels: int, with_z: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = _group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = _group_norm(out_channels)
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        self.act = nn.GELU()
        self.z_branch = ZResidual(out_channels) if with_z else None

    def forward(self, x: torch.Tensor, n_z: int) -> torch.Tensor:
        y = self.act(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        y = self.act(y + self.shortcut(x))
        if self.z_branch is not None:
            y = self.z_branch(y, n_z)
        return y


class NeXtBlock(nn.Module):
    """A ConvNeXt block in xy (7x7 depthwise, per-slice norm, inverted bottleneck), then an optional z module.

    Args:
        channels: The number of channels.
        z_mode: The z module: 'attention' (`ZAttention`), 'conv' (`ZResidual`) or None.
    """
    def __init__(self, channels: int, z_mode: Optional[str] = None):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.GroupNorm(1, channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, 4 * channels, 1), nn.GELU(), nn.Conv2d(4 * channels, channels, 1),
        )
        if z_mode == "attention":
            self.z_module = ZAttention(channels)
        elif z_mode == "conv":
            self.z_module = ZResidual(channels)
        elif z_mode is None:
            self.z_module = None
        else:
            raise ValueError(f"Invalid z_mode {z_mode}")

    def forward(self, x: torch.Tensor, n_z: int) -> torch.Tensor:
        x = x + self.mlp(self.norm(self.depthwise(x)))
        return x if self.z_module is None else self.z_module(x, n_z)


#
# DECODERS
#


class HybridDecoder(nn.Module):
    """Base class of the hybrid decoders.

    It holds the projections of the stride-4 and stride-8 encoder features and the image stem, which computes
    the skips at stride 1 (`stem1`) and stride 2 (`stem2`). A decoder implements
    `forward(features, image, n_z)`, with all tensors folded:
    - features: The encoder features at stride 4, 8 and 16, each of shape (B * Z, C_l, H_l, W_l).
    - image: The preprocessed image of shape (B * Z, 3, H, W).
    - n_z: The number of slices Z.
    It returns the output before the activation, of shape (B * Z, out_channels, H, W).

    Args:
        out_channels: The number of output channels.
        stem_widths: The widths of the image stem at strides 1 and 2.
        in_channels: The channels of the encoder features at stride 4, 8 and 16.
        skip_projections: The output channels of the 1x1 projections of the stride-4 and stride-8 features.
            The defaults correspond to `conv_s0` and `conv_s1` of the SAM2 mask decoder.
            If None, the decoder uses the features without projections.
        norm: The function that creates the normalization layer of the image stem for a number of channels.
        activation: The activation of the image stem.
    """
    def __init__(
        self,
        out_channels: int,
        stem_widths: Sequence[int] = (16, 32),
        in_channels: Sequence[int] = (256, 256, 256),
        skip_projections: Optional[Sequence[int]] = (32, 64),
        norm: Callable[[int], nn.Module] = _group_norm,
        activation: Callable[[], nn.Module] = nn.GELU,
    ):
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError(f"Expected the channels of three feature levels, got {in_channels}.")
        self.out_channels = out_channels
        self.in_channels = tuple(in_channels)
        st1, st2 = stem_widths
        self.stem1 = nn.Sequential(nn.Conv2d(3, st1, 3, padding=1, bias=False), norm(st1), activation())
        self.stem2 = nn.Sequential(nn.Conv2d(st1, st2, 3, stride=2, padding=1, bias=False), norm(st2), activation())
        if skip_projections is None:
            self.proj_s4, self.proj_s8 = None, None
            self.skip_channels = self.in_channels[:2]
        else:
            self.skip_channels = tuple(skip_projections)
            self.proj_s4 = nn.Conv2d(self.in_channels[0], self.skip_channels[0], 1)
            self.proj_s8 = nn.Conv2d(self.in_channels[1], self.skip_channels[1], 1)

    def _project(self, features: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s4, s8, s16 = features
        if self.proj_s4 is not None:
            s4, s8 = self.proj_s4(s4), self.proj_s8(s8)
        return s4, s8, s16


class HieraUDecoder(HybridDecoder):
    """A plain skip U-decoder on the encoder features at stride 4, 8 and 16 and an image stem.

    All spatial operations are 2D on the slices folded into the batch. The z branches of the blocks at
    strides 16 to 2 carry the 3D context. The full-resolution block is 2D only.

    Args:
        out_channels: The number of output channels.
        widths: The decoder widths at strides 16, 8, 4, 2 and 1.
        stem_widths: The widths of the image stem at strides 1 and 2.
        in_channels: The channels of the encoder features at stride 4, 8 and 16.
        skip_projections: The output channels of the 1x1 projections of the stride-4 and stride-8 features.
            If None, the decoder uses the features without projections.
    """
    def __init__(
        self,
        out_channels: int = 5,
        widths: Sequence[int] = (192, 128, 64, 32, 16),
        stem_widths: Sequence[int] = (16, 32),
        in_channels: Sequence[int] = (256, 256, 256),
        skip_projections: Optional[Sequence[int]] = (32, 64),
    ):
        super().__init__(out_channels, stem_widths, in_channels, skip_projections)
        w16, w8, w4, w2, w1 = widths
        st1, st2 = stem_widths
        c4, c8 = self.skip_channels
        self.block16 = HybridBlock(self.in_channels[2], w16)
        self.block8 = HybridBlock(w16 + c8, w8)
        self.block4 = HybridBlock(w8 + c4, w4)
        self.block2 = HybridBlock(w4 + st2, w2)
        self.block1 = HybridBlock(w2 + st1, w1, with_z=False)
        self.out_conv = nn.Conv2d(w1, out_channels, 1)

    def forward(self, features: Sequence[torch.Tensor], image: torch.Tensor, n_z: int) -> torch.Tensor:
        """Apply the decoder, see `HybridDecoder`."""
        s4, s8, s16 = self._project(features)
        stem1 = self.stem1(image)
        stem2 = self.stem2(stem1)
        x = self.block16(s16, n_z)
        x = self.block8(_resize_cat(x, s8), n_z)
        x = self.block4(_resize_cat(x, s4), n_z)
        x = self.block2(_resize_cat(x, stem2), n_z)
        x = self.block1(_resize_cat(x, stem1), n_z)
        return self.out_conv(x)


class C3F2Decoder(HybridDecoder):
    """A coarse-3D / fine-2D fusion head on the encoder features at stride 4, 8 and 16 and an image stem.

    Stride 16: two ConvNeXt blocks, each followed by attention along z. All long-range z reasoning happens here.
    Stride 8 and 4: additive top-down fusion with the lateral features and ConvNeXt blocks with z convolutions
    (the last block at stride 4 is 2D). Finer: two transposed convolutions to full resolution, each fused with the
    image stem, in 2D only.

    Args:
        out_channels: The number of output channels.
        widths: The widths at strides 16, 8 and 4.
        head_width: The width of the stride-2 head. The stride-1 head uses the stride-1 stem width.
        stem_widths: The widths of the image stem at strides 1 and 2.
        in_channels: The channels of the encoder features at stride 4, 8 and 16.
        skip_projections: The output channels of the 1x1 projections of the stride-4 and stride-8 features.
            If None, the decoder uses the features without projections.
    """
    def __init__(
        self,
        out_channels: int = 5,
        widths: Sequence[int] = (256, 128, 64),
        head_width: int = 32,
        stem_widths: Sequence[int] = (16, 32),
        in_channels: Sequence[int] = (256, 256, 256),
        skip_projections: Optional[Sequence[int]] = (32, 64),
    ):
        super().__init__(out_channels, stem_widths, in_channels, skip_projections)
        w16, w8, w4 = widths
        st1, st2 = stem_widths
        c4, c8 = self.skip_channels
        self.lateral16 = nn.Conv2d(self.in_channels[2], w16, 1)
        self.blocks16 = nn.ModuleList([NeXtBlock(w16, "attention"), NeXtBlock(w16, "attention")])
        self.lateral8 = nn.Conv2d(c8, w8, 1)
        self.reduce8 = nn.Conv2d(w16, w8, 1)
        self.blocks8 = nn.ModuleList([NeXtBlock(w8, "conv"), NeXtBlock(w8, "conv")])
        self.lateral4 = nn.Conv2d(c4, w4, 1)
        self.reduce4 = nn.Conv2d(w8, w4, 1)
        self.blocks4 = nn.ModuleList([NeXtBlock(w4, "conv"), NeXtBlock(w4, None)])
        self.up2 = nn.Sequential(nn.ConvTranspose2d(w4, head_width, 2, stride=2), _group_norm(head_width), nn.GELU())
        self.fuse2 = nn.Sequential(
            nn.Conv2d(head_width + st2, head_width, 3, padding=1, bias=False), _group_norm(head_width), nn.GELU(),
        )
        self.up1 = nn.Sequential(nn.ConvTranspose2d(head_width, st1, 2, stride=2), _group_norm(st1), nn.GELU())
        self.fuse1 = nn.Sequential(
            nn.Conv2d(2 * st1, st1, 3, padding=1, bias=False), _group_norm(st1), nn.GELU(),
        )
        self.out_conv = nn.Conv2d(st1, out_channels, 1)

    @staticmethod
    def _run(blocks, x, n_z):
        for block in blocks:
            x = block(x, n_z)
        return x

    def forward(self, features: Sequence[torch.Tensor], image: torch.Tensor, n_z: int) -> torch.Tensor:
        """Apply the decoder, see `HybridDecoder`."""
        s4, s8, s16 = self._project(features)
        stem1 = self.stem1(image)
        stem2 = self.stem2(stem1)
        x = self._run(self.blocks16, self.lateral16(s16), n_z)
        lateral = self.lateral8(s8)
        x = self._run(self.blocks8, _resize_to(self.reduce8(x), lateral) + lateral, n_z)
        lateral = self.lateral4(s4)
        x = self._run(self.blocks4, _resize_to(self.reduce4(x), lateral) + lateral, n_z)
        x = self.fuse2(_resize_cat(self.up2(x), stem2))
        x = self.fuse1(_resize_cat(self.up1(x), stem1))
        return self.out_conv(x)


#
# CAGD-3D: MedCAGD adapted to the hybrid 2D/3D recipe.
#


def _cagd_group_norm(channels: int) -> nn.GroupNorm:
    """GroupNorm with the group count of MedCAGD (4 to 32 channels per group). Per slice on folded inputs."""
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0 and 4 <= channels // groups <= 32:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class ECA(nn.Module):
    """Efficient channel attention of MedCAGD with a learned temperature, pooled per slice.

    MedCAGD averages three convolutions of the global mean (its pooling scales 1, 2, 4 all reduce to the global
    mean). This is linear and thus equal to one convolution, which is used here.

    Args:
        channels: The number of channels.
        is_encoder: Whether to use the kernel size rule of MedCAGD for encoder (skip) or for decoder features.
    """
    def __init__(self, channels: int, is_encoder: bool = True):
        super().__init__()
        gamma, b = (2.0, 1.0) if is_encoder else (2.4, 0.8)
        k = int(abs(math.log2(channels) / gamma + b))
        k = k + 1 if k % 2 == 0 else k
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.log_tau = nn.Parameter(torch.zeros(()))
        self.tau_min, self.tau_max = 0.5, 2.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x.mean(dim=(2, 3))[:, None])[:, 0]  # (N, C), N = slices
        tau = self.log_tau.exp().clamp(self.tau_min, self.tau_max)
        attention = torch.sigmoid(logits / tau)
        attention = attention / (attention.mean(dim=1, keepdim=True) + 1e-6)
        return x * attention[..., None, None]


class GlobalContext(nn.Module):
    """The residual global-context attention of MedCAGD (a GCNet block), pooled per slice.

    Args:
        channels: The number of channels.
        reduction: The channel reduction of the transform.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        inter = max(4, channels // reduction)  # MedCAGD: max(1, ...); same for the default widths (>= 64)
        self.mask = nn.Conv2d(channels, 1, kernel_size=1)
        self.transform = nn.Sequential(
            nn.Conv2d(channels, inter, kernel_size=1, bias=False),
            nn.GroupNorm(1, inter),  # = LayerNorm([inter, 1, 1]) of MedCAGD
            nn.ReLU(inplace=True),
            nn.Conv2d(inter, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        weights = torch.softmax(self.mask(x).reshape(n, 1, h * w), dim=-1)
        context = torch.bmm(x.reshape(n, c, h * w), weights.transpose(1, 2)).reshape(n, c, 1, 1)
        return x + self.transform(context)


class SCAGate(nn.Module):
    """The attention gate of MedCAGD (SCA-Gate): gates the skip features with guidance from the decoder.

    Changed for large inputs: the spatial competition is a softmax over the H * W positions of a slice, with
    values of about 1 / (H * W), which is numerically zero for inputs of size 1024. It is rescaled by H * W
    (mean one) and clamped at `max_spatial`.

    Args:
        guide_channels: The channels of the upsampled decoder features.
        skip_channels: The channels of the skip features.
        inter_channels: The latent width.
        max_spatial: The maximum of the rescaled spatial attention.
    """
    def __init__(self, guide_channels: int, skip_channels: int, inter_channels: int, max_spatial: float = 4.0):
        super().__init__()
        self.eca_skip = ECA(skip_channels, is_encoder=True)
        self.eca_guide = ECA(guide_channels, is_encoder=False)

        def branch(channels):
            return nn.Sequential(
                nn.Conv2d(channels, inter_channels, 3, padding=1, bias=False), _cagd_group_norm(inter_channels),
                nn.SiLU(),
                nn.Conv2d(inter_channels, inter_channels, 3, padding=2, dilation=2, groups=inter_channels,
                          bias=False),
                _cagd_group_norm(inter_channels), nn.SiLU(),
            )
        self.theta, self.phi = branch(guide_channels), branch(skip_channels)
        self.global_fc = nn.Sequential(
            nn.Conv2d(guide_channels + skip_channels, inter_channels, 1), nn.SiLU(), nn.Sigmoid(),
        )
        self.dw3 = nn.Conv2d(inter_channels, inter_channels, 3, padding=1, groups=inter_channels, bias=False)
        self.dw5 = nn.Conv2d(inter_channels, inter_channels, 5, padding=2, groups=inter_channels, bias=False)
        self.score = nn.Conv2d(inter_channels, 1, 1, bias=False)
        self.log_tau = nn.Parameter(torch.tensor(0.7).log())
        self.tau_min, self.tau_max = 0.3, 2.0
        self.max_spatial = max_spatial
        self.att = nn.Sequential(nn.GroupNorm(1, inter_channels), nn.Conv2d(inter_channels, 1, 1), nn.Hardsigmoid())

    def forward(self, guide: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        skip = self.eca_skip(skip)
        guide = self.eca_guide(guide)
        fused = F.normalize(self.theta(guide) * self.phi(skip), p=2, dim=1)
        fused = fused + self.global_fc(torch.cat([guide, skip], dim=1).mean(dim=(2, 3), keepdim=True)) * fused
        score = self.score(self.dw3(fused) + self.dw5(fused))
        n, _, h, w = score.shape
        tau = self.log_tau.exp().clamp(self.tau_min, self.tau_max)
        spatial = torch.softmax(score.reshape(n, h * w) / tau, dim=-1).reshape(n, 1, h, w) * (h * w)
        fused = fused * (1.0 + spatial.clamp(max=self.max_spatial))
        return skip * self.att(fused + 1e-5)


class DWBlock(nn.Module):
    """The conv block of MedCAGD (depthwise 3x3 and pointwise, twice), followed by a zero-initialized z branch.

    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        with_z: Whether to add the z branch.
    """
    def __init__(self, in_channels: int, out_channels: int, with_z: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            _cagd_group_norm(in_channels), nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 1, bias=False), _cagd_group_norm(out_channels), nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
            _cagd_group_norm(out_channels), nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 1, bias=False), _cagd_group_norm(out_channels), nn.SiLU(),
        )
        self.z_branch = ZResidual(out_channels) if with_z else None

    def forward(self, x: torch.Tensor, n_z: int) -> torch.Tensor:
        x = self.block(x)
        return x if self.z_branch is None else self.z_branch(x, n_z)


class RefinementBlock(nn.Module):
    """The refinement block of MedCAGD: conv block, optional depthwise spatial conv and channel recalibration.

    Args:
        channels: The number of channels.
        spatial_focus: Whether to add the depthwise spatial convolution.
        with_z: Whether to add the z branch to the conv block.
    """
    def __init__(self, channels: int, spatial_focus: bool, with_z: bool = True):
        super().__init__()
        self.conv = DWBlock(channels, channels, with_z=with_z)
        self.spatial = nn.Conv2d(channels, channels, 3, padding=1, groups=channels) if spatial_focus else None
        self.eca = ECA(channels)

    def forward(self, x: torch.Tensor, n_z: int) -> torch.Tensor:
        x = self.conv(x, n_z)
        if self.spatial is not None:
            x = self.spatial(x)
        return self.eca(x)


class CAGDStage(nn.Module):
    """The decoder block of MedCAGD: transposed-conv upsampling, gated skip fusion and the conv block.

    The skips are gated only if they have more than 64 channels, as in MedCAGD.

    Args:
        in_channels: The number of input channels.
        skip_channels: The channels of the skip features.
        out_channels: The number of output channels.
        with_z: Whether to add the z branch to the conv block.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, with_z: bool = True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.gate = SCAGate(out_channels, skip_channels, max(1, skip_channels // 2)) if skip_channels > 64 else None
        self.conv = DWBlock(out_channels + skip_channels, out_channels, with_z=with_z)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, n_z: int) -> torch.Tensor:
        x = _resize_to(self.up(x), skip)
        if self.gate is not None:
            skip = self.gate(x, skip)
        return self.conv(torch.cat([x, skip], dim=1), n_z)


class ContextAggregator(nn.Module):
    """The context aggregator of MedCAGD: all pyramid levels projected, resized and averaged, plus global context.

    Args:
        in_channels: The channels of the pyramid levels.
        out_channels: The number of output channels.
    """
    def __init__(self, in_channels: Sequence[int], out_channels: int):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels])
        self.global_context = GlobalContext(out_channels)

    def forward(self, features: Sequence[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
        fused = sum(_resize_to(proj(f), reference) for f, proj in zip(features, self.proj))
        return self.global_context(fused / len(features))


class CAGD3DDecoder(HybridDecoder):
    """The decoder of MedCAGD on the encoder features at stride 4, 8 and 16, adapted to the hybrid 2D/3D recipe.

    MedCAGD: Wazir et al., arXiv 2607.00409, https://github.com/saadwazir/MedCAGD.

    Pyramid: 1x1 projections of the features at stride 4, 8 and 16. The stride-32 level is a strided 3x3 convolution
    of the stride-16 features. The bottleneck (conv, attention along z, global context) and the stages at strides
    16, 8 and 4 (gated fusion, context aggregator, refinement) follow MedCAGD. All global pooling is per slice
    and the blocks have zero-initialized z branches.

    Changed against MedCAGD: the last stage runs at stride 2 with an image-stem skip instead of at full resolution
    (the full-resolution stage of MedCAGD costs 1.7x the FLOPs). It is followed by a transposed convolution to
    full resolution, fused with the stride-1 stem.

    Deep supervision (optional): 1x1 heads on the stages at strides 16, 8 and 4 predict all output channels.
    In training mode their outputs, upsampled to the image size as in MedCAGD, are kept in `aux_outputs` (deepest
    first; MedCAGD weights them with 0.4, 0.3 and 0.2). `HybridUNETR` includes them in its forward result.
    If it is enabled, the loss must use these outputs, otherwise the heads do not receive gradients.

    Args:
        out_channels: The number of output channels.
        widths: The pyramid widths at strides 4, 8, 16 and 32.
        stem_widths: The widths of the image stem at strides 1 and 2.
        deep_supervision: Whether to build the deep supervision heads.
        in_channels: The channels of the encoder features at stride 4, 8 and 16.
        skip_projections: The output channels of the 1x1 projections of the stride-4 and stride-8 features.
            If None, the decoder uses the features without projections.
    """
    def __init__(
        self,
        out_channels: int = 5,
        widths: Sequence[int] = (64, 128, 320, 512),
        stem_widths: Sequence[int] = (16, 32),
        deep_supervision: bool = False,
        in_channels: Sequence[int] = (256, 256, 256),
        skip_projections: Optional[Sequence[int]] = (32, 64),
    ):
        super().__init__(
            out_channels, stem_widths, in_channels, skip_projections, norm=_cagd_group_norm, activation=nn.SiLU,
        )
        w4, w8, w16, w32 = widths
        st1, st2 = stem_widths
        c4, c8 = self.skip_channels
        c16 = self.in_channels[2]
        self.p4 = nn.Conv2d(c4, w4, 1)
        self.p8 = nn.Conv2d(c8, w8, 1)
        self.p16 = nn.Conv2d(c16, w16, 1)
        self.p32 = nn.Conv2d(c16, w32, 3, stride=2, padding=1)

        self.bottleneck_conv = nn.Sequential(nn.Conv2d(w32, w32, 3, padding=1), _cagd_group_norm(w32), nn.SiLU())
        self.bottleneck_z = ZAttention(w32)
        self.bottleneck_context = GlobalContext(w32)

        levels = (w4, w8, w16, w32)
        self.stage16, self.context16, self.ref16 = (
            CAGDStage(w32, w16, w16), ContextAggregator(levels, w16), RefinementBlock(w16, spatial_focus=False),
        )
        self.stage8, self.context8, self.ref8 = (
            CAGDStage(w16, w8, w8), ContextAggregator(levels, w8), RefinementBlock(w8, spatial_focus=True),
        )
        self.stage4, self.context4, self.ref4 = (
            CAGDStage(w8, w4, w4), ContextAggregator(levels, w4), RefinementBlock(w4, spatial_focus=True),
        )
        self.stage2 = CAGDStage(w4, st2, st2)
        self.ref2 = RefinementBlock(st2, spatial_focus=True)

        self.head_up = nn.Sequential(
            nn.ConvTranspose2d(st2, st1, kernel_size=2, stride=2), _cagd_group_norm(st1), nn.SiLU(),
        )
        self.head_fuse = nn.Sequential(
            nn.Conv2d(2 * st1, st1, 3, padding=1, bias=False), _cagd_group_norm(st1), nn.SiLU(),
        )
        self.out_conv = nn.Conv2d(st1, out_channels, 1)
        self.aux_heads = nn.ModuleList(
            [nn.Conv2d(c, out_channels, 1) for c in (w16, w8, w4)]
        ) if deep_supervision else None
        self.aux_outputs: Optional[List[torch.Tensor]] = None

    def train(self, mode: bool = True):
        """Set the training mode. The deep supervision outputs are dropped when leaving the training mode."""
        if not mode:
            self.aux_outputs = None
        return super().train(mode)

    def forward(self, features: Sequence[torch.Tensor], image: torch.Tensor, n_z: int) -> torch.Tensor:
        """Apply the decoder, see `HybridDecoder`."""
        s4, s8, s16 = self._project(features)
        stem1 = self.stem1(image)
        stem2 = self.stem2(stem1)
        p4, p8, p16 = self.p4(s4), self.p8(s8), self.p16(s16)
        b = self.bottleneck_conv(self.p32(s16))
        b = self.bottleneck_context(self.bottleneck_z(b, n_z))
        levels = [p4, p8, p16, b]

        d16 = self.stage16(b, p16, n_z)
        d16 = self.ref16(d16 + self.context16(levels, d16), n_z)
        d8 = self.stage8(d16, p8, n_z)
        d8 = self.ref8(d8 + self.context8(levels, d8), n_z)
        d4 = self.stage4(d8, p4, n_z)
        d4 = self.ref4(d4 + self.context4(levels, d4), n_z)
        d2 = self.ref2(self.stage2(d4, stem2, n_z), n_z)

        out = self.out_conv(self.head_fuse(_resize_cat(self.head_up(d2), stem1)))

        self.aux_outputs = None
        if self.training and self.aux_heads is not None:
            self.aux_outputs = [
                _resize_to(head(d), image) for head, d in zip(self.aux_heads, (d16, d8, d4))
            ]
        return out


# The hybrid decoders by name. To add a decoder, implement the `HybridDecoder` interface and register it here.
# A decoder with deep supervision keeps the auxiliary outputs of the last forward pass in training mode in the
# attribute `aux_outputs`. `HybridUNETR.decode` processes them like the main output.
HYBRID_DECODERS = {"hiera_u": HieraUDecoder, "c3f2": C3F2Decoder, "cagd_3d": CAGD3DDecoder}
"""The hybrid decoders by name."""


#
# MODEL
#


class HybridUNETR(UNETRBase):
    """A UNETR for 2D and 3D inputs with a multi-scale encoder and a hybrid 2D/3D decoder.

    The model applies the encoder to each slice and decodes all slices together. The same weights process
    images of shape (B, C, Y, X) and volumes of shape (B, C, Z, Y, X). An image is processed like a volume
    with a single slice.

    The state of the model has the two prefixes 'encoder.' and 'decoder.'.

    Args:
        img_size: The size of the input for the image encoder. Input images will be resized to match this size.
        backbone: The name of the vision transformer implementation. The model was designed for 'sam2'.
        encoder: The image encoder. Can either be a name, such as 'hvit_t', or a torch module.
            A module must have the attribute `img_size` and return a tuple, with the list of features
            at stride 4, 8 and 16 as second element, like `torch_em.model.vit.ViT_Sam2`.
        decoder: The decoder. Can either be a name from `HYBRID_DECODERS` ('hiera_u', 'c3f2' or 'cagd_3d') or
            a torch module that implements the interface of `HybridDecoder` and has the attribute `in_channels`.
        decoder_kwargs: The keyword arguments for the decoder, if it is given by name.
        out_channels: The number of output channels. It is ignored if the decoder is a torch module.
        use_sam_stats: Whether to normalize the input data with the statistics of the pretrained SAM2 model.
        use_mae_stats: Whether to normalize the input data with the statistics of the pretrained MAE model.
        use_dino_stats: Whether to normalize the input data with the statistics of the pretrained DINO models.
        use_imagenet_stats: Whether to normalize with standard ImageNet statistics.
        resize_input: Whether to resize the input images to match `img_size`.
        encoder_checkpoint: Checkpoint for initializing the image encoder.
            Can either be a filepath or an already loaded checkpoint. It does not initialize the projections
            of the stride-4 and stride-8 features in the decoder, see `init_skip_projections`.
        final_activation: The activation to apply to the output.
        perform_range_checks: Whether to validate the input value range before normalization on each forward pass.
        kwargs: Keyword arguments for `torch_em.model.vit.get_vision_transformer`, if the encoder is given by name.
    """
    def __init__(
        self,
        img_size: int = 1024,
        backbone: str = "sam2",
        encoder: Union[nn.Module, str] = "hvit_t",
        decoder: Union[nn.Module, str] = "hiera_u",
        decoder_kwargs: Optional[dict] = None,
        out_channels: int = 1,
        use_sam_stats: bool = False,
        use_mae_stats: bool = False,
        use_dino_stats: bool = False,
        use_imagenet_stats: bool = False,
        resize_input: bool = True,
        encoder_checkpoint: Optional[Union[str, PathLike, Mapping]] = None,
        final_activation: Optional[Union[str, nn.Module]] = None,
        perform_range_checks: bool = True,
        **kwargs
    ) -> None:
        if isinstance(decoder, str):
            if decoder not in HYBRID_DECODERS:
                raise ValueError(f"Invalid decoder '{decoder}', choose from {list(HYBRID_DECODERS)}.")
            decoder_module = HYBRID_DECODERS[decoder](out_channels=out_channels, **(decoder_kwargs or {}))
        elif isinstance(decoder, nn.Module):
            if decoder_kwargs:
                raise ValueError("The decoder_kwargs are only supported for a decoder that is given by name.")
            decoder_module = decoder
        else:
            raise TypeError("The decoder must be a decoder name or a torch module.")

        super().__init__(
            img_size=img_size,
            backbone=backbone,
            encoder=encoder,
            out_channels=out_channels,
            use_sam_stats=use_sam_stats,
            use_mae_stats=use_mae_stats,
            use_dino_stats=use_dino_stats,
            use_imagenet_stats=use_imagenet_stats,
            resize_input=resize_input,
            encoder_checkpoint=encoder_checkpoint,
            final_activation=final_activation,
            use_skip_connection=True,
            # The stride-16 feature width of the decoder. This avoids deriving it from the encoder,
            # which is not possible for arbitrary encoder modules.
            embed_dim=decoder_module.in_channels[-1],
            perform_range_checks=perform_range_checks,
            **kwargs,
        )
        self.init_kwargs = {
            "img_size": img_size, "backbone": backbone, "encoder": encoder, "decoder": decoder,
            "decoder_kwargs": decoder_kwargs, "out_channels": out_channels, "use_sam_stats": use_sam_stats,
            "use_mae_stats": use_mae_stats, "use_dino_stats": use_dino_stats,
            "use_imagenet_stats": use_imagenet_stats, "resize_input": resize_input,
            "final_activation": final_activation, "perform_range_checks": perform_range_checks, **kwargs,
        }

        self.decoder = decoder_module
        self.out_channels = getattr(decoder_module, "out_channels", out_channels)
        # The deep supervision outputs of the last forward pass in training mode, for decoders that have them
        # ('cagd_3d' with deep supervision). They are activated and resized like the main output. None otherwise.
        self.auxiliary_outputs: Optional[List[torch.Tensor]] = None

    def train(self, mode: bool = True):
        """Set the training mode. The deep supervision outputs are dropped when leaving the training mode."""
        if not mode:
            self.auxiliary_outputs = None
        return super().train(mode)

    def init_skip_projections(self, conv_s0: nn.Module, conv_s1: nn.Module) -> None:
        """Initialize the projections of the stride-4 and stride-8 features in the decoder from other modules.

        This copies the weights, the modules are not shared. For SAM2, pass `sam_mask_decoder.conv_s0` and
        `sam_mask_decoder.conv_s1` of the SAM2 model.

        Args:
            conv_s0: The projection of the stride-4 features.
            conv_s1: The projection of the stride-8 features.
        """
        proj_s4, proj_s8 = getattr(self.decoder, "proj_s4", None), getattr(self.decoder, "proj_s8", None)
        if proj_s4 is None or proj_s8 is None:
            raise RuntimeError("The decoder does not have projections for the stride-4 and stride-8 features.")
        proj_s4.load_state_dict(conv_s0.state_dict())
        proj_s8.load_state_dict(conv_s1.state_dict())

    @property
    def decoder_config(self) -> dict:
        """The arguments that rebuild the decoder. They can be serialized to JSON, e.g. to store them in checkpoints.
        """
        decoder = self.init_kwargs["decoder"]
        if not isinstance(decoder, str):
            raise RuntimeError("The decoder_config is only available for a decoder that is given by name.")
        return {
            "decoder": decoder, "decoder_kwargs": dict(self.init_kwargs["decoder_kwargs"] or {}),
            "out_channels": self.out_channels,
        }

    def encode(self, x: torch.Tensor) -> Tuple[Dict[str, Union[List[torch.Tensor], torch.Tensor, int]], Tuple]:
        """Preprocess the input and run the image encoder on all slices.

        Args:
            x: The input tensor with shape (B, C, Z, Y, X) or (B, C, Y, X).

        Returns:
            The features to pass to `decode`: a dictionary with the encoder features at stride 4, 8 and 16
            ('features'), the preprocessed image ('image'), both with z folded into the batch, and the
            number of slices ('n_z').
            The spatial shape (Z, Y, X) after preprocessing.
        """
        if x.ndim == 4:  # An image is processed as a volume with a single slice.
            x = x[:, :, None]
        elif x.ndim != 5:
            raise ValueError(f"Expected 4d or 5d inputs, got {tuple(x.shape)}.")

        n_z = x.shape[2]
        # The slices are preprocessed as images, so that images and volumes are resized in the same way.
        image, input_shape = self.preprocess(fold_z(x))
        input_shape = (n_z, *input_shape)
        _, levels = self._encode_image(image)
        if levels is None or len(levels) != 3:
            raise ValueError(
                "The encoder must return a tuple with the list of features at stride 4, 8 and 16 as second element."
            )
        return {"features": list(levels), "image": image, "n_z": n_z}, input_shape

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Predict masks for an image or volume.

        During training with deep supervision, return `(prediction, auxiliary_outputs)`.
        The training loss should use the tensors from this pair, including under DistributedDataParallel.
        Otherwise, return the prediction tensor.
        """
        prediction = super().forward(x)
        if self.auxiliary_outputs is not None:
            return prediction, self.auxiliary_outputs
        return prediction

    def decode(self, features, input_shape: Tuple[int, ...], original_shape: Tuple[int, ...]) -> torch.Tensor:
        """Run the decoder on the encoder features.

        Args:
            features: The dictionary returned by `encode`.
            input_shape: The spatial shape (Z, Y, X) after preprocessing, returned by `encode`.
            original_shape: The spatial shape of the original input, (Z, Y, X) or (Y, X).

        Returns:
            The output, resized to `original_shape`. It has no z axis if `original_shape` is (Y, X).
            In training mode, the deep supervision outputs of the decoder are stored in `auxiliary_outputs`.
        """
        is_2d = len(original_shape) == 2
        if is_2d:
            original_shape = (1, *original_shape)

        depth, height, width = input_shape
        # Crop on the same grid that preprocessing padded the slices to.
        padded_shape = (depth, *features["image"].shape[-2:])

        def _postprocess(x):
            x = unfold_z(x, n_z)
            if self.final_activation is not None:
                x = self.final_activation(x)
            x = F.interpolate(x, padded_shape, mode="trilinear", align_corners=False)
            x = x[..., :depth, :height, :width]
            x = F.interpolate(x, original_shape, mode="trilinear", align_corners=False)
            return x[:, :, 0] if is_2d else x

        n_z = features["n_z"]
        x = self.decoder(features["features"], features["image"], n_z)

        auxiliary_outputs = getattr(self.decoder, "aux_outputs", None)
        if self.training and auxiliary_outputs:
            self.auxiliary_outputs = [_postprocess(aux) for aux in auxiliary_outputs]
        else:
            self.auxiliary_outputs = None

        return _postprocess(x)
