"""UNETR with hybrid 2D/3D decoders that read multi-scale encoder features.

The decoders read the encoder features at stride 4, 8 and 16 (e.g. of the SAM2 Hiera encoder) and the preprocessed
image, from which an image stem computes the skips at stride 2 and 1. They follow the hybrid 2D/3D recipe:

- All spatial operations are 2D and run on the z-slices folded into the batch: (B, C, Z, H, W) becomes
  (B * Z, C, H, W). The normalization is a GroupNorm on the folded tensor, so its statistics are per slice.
- The context along z comes only from residual branches that are zero-initialized and skipped for a single slice:
  `ZResidual` (convolution along z) and `ZAttention` (attention along z).

The same weights therefore decode single images (Z = 1) and volumes (Z > 1). At initialization a volume decodes
exactly to the stack of its decoded slices.

The decoders:
- `HieraUDecoder` ("hiera_u"): a plain skip U-decoder.
- `C3F2Decoder` ("c3f2"): a coarse-3D / fine-2D fusion head. ConvNeXt blocks with attention along z at stride 16,
  with convolutions along z at strides 8 and 4, and a 2D image-guided upsampling head.

`HybridUNETR` combines an encoder with one of the decoders.
"""

from collections.abc import Mapping
from os import PathLike
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unetr import UNETRBase, _load_checkpoint_state


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
        """Apply the branch.

        Args:
            x: The features with z folded into the batch, (B * Z, C, H, W).
            n_z: The number of slices Z.

        Returns:
            The features with the same shape.
        """
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
        """Apply the attention.

        Args:
            x: The features with z folded into the batch, (B * Z, C, H, W).
            n_z: The number of slices Z.

        Returns:
            The features with the same shape.
        """
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
        """Apply the block.

        Args:
            x: The features with z folded into the batch, (B * Z, C, H, W).
            n_z: The number of slices Z.

        Returns:
            The output features.
        """
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
        """Apply the block.

        Args:
            x: The features with z folded into the batch, (B * Z, C, H, W).
            n_z: The number of slices Z.

        Returns:
            The features with the same shape.
        """
        x = x + self.mlp(self.norm(self.depthwise(x)))
        return x if self.z_module is None else self.z_module(x, n_z)


#
# DECODERS
#


class HybridDecoder(nn.Module):
    """Base class of the hybrid decoders.

    It holds the projections of the stride-4 and stride-8 encoder features. A decoder implements
    `forward(features, image, n_z)`, with all tensors folded:
    - features: The encoder features at stride 4, 8 and 16, each of shape (B * Z, C_l, H_l, W_l).
    - image: The preprocessed image of shape (B * Z, 3, H, W).
    - n_z: The number of slices Z.
    It returns the output before the activation, of shape (B * Z, out_channels, H, W).

    Args:
        out_channels: The number of output channels.
        in_channels: The channels of the encoder features at stride 4, 8 and 16.
        skip_projections: The output channels of the 1x1 projections of the stride-4 and stride-8 features.
            The defaults correspond to `conv_s0` and `conv_s1` of the SAM2 mask decoder.
            If None, the decoder uses the features without projections.
    """
    def __init__(
        self,
        out_channels: int,
        in_channels: Sequence[int] = (256, 256, 256),
        skip_projections: Optional[Sequence[int]] = (32, 64),
    ):
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError(f"Expected the channels of three feature levels, got {in_channels}.")
        self.out_channels = out_channels
        self.in_channels = tuple(in_channels)
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
        super().__init__(out_channels, in_channels, skip_projections)
        w16, w8, w4, w2, w1 = widths
        st1, st2 = stem_widths
        c4, c8 = self.skip_channels
        self.stem1 = nn.Sequential(nn.Conv2d(3, st1, 3, padding=1, bias=False), _group_norm(st1), nn.GELU())
        self.stem2 = nn.Sequential(
            nn.Conv2d(st1, st2, 3, stride=2, padding=1, bias=False), _group_norm(st2), nn.GELU(),
        )
        self.block16 = HybridBlock(self.in_channels[2], w16)
        self.block8 = HybridBlock(w16 + c8, w8)
        self.block4 = HybridBlock(w8 + c4, w4)
        self.block2 = HybridBlock(w4 + st2, w2)
        self.block1 = HybridBlock(w2 + st1, w1, with_z=False)
        self.out_conv = nn.Conv2d(w1, out_channels, 1)

    @staticmethod
    def _up(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, features: Sequence[torch.Tensor], image: torch.Tensor, n_z: int) -> torch.Tensor:
        """Apply the decoder.

        Args:
            features: The encoder features at stride 4, 8 and 16 with z folded into the batch.
            image: The preprocessed image with z folded into the batch, (B * Z, 3, H, W).
            n_z: The number of slices Z.

        Returns:
            The decoder output before the activation, (B * Z, out_channels, H, W).
        """
        s4, s8, s16 = self._project(features)
        stem1 = self.stem1(image)
        stem2 = self.stem2(stem1)
        x = self.block16(s16, n_z)
        x = self.block8(self._up(x, s8), n_z)
        x = self.block4(self._up(x, s4), n_z)
        x = self.block2(self._up(x, stem2), n_z)
        x = self.block1(self._up(x, stem1), n_z)
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
        super().__init__(out_channels, in_channels, skip_projections)
        w16, w8, w4 = widths
        st1, st2 = stem_widths
        c4, c8 = self.skip_channels
        self.stem1 = nn.Sequential(nn.Conv2d(3, st1, 3, padding=1, bias=False), _group_norm(st1), nn.GELU())
        self.stem2 = nn.Sequential(
            nn.Conv2d(st1, st2, 3, stride=2, padding=1, bias=False), _group_norm(st2), nn.GELU(),
        )
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
        """Apply the decoder.

        Args:
            features: The encoder features at stride 4, 8 and 16 with z folded into the batch.
            image: The preprocessed image with z folded into the batch, (B * Z, 3, H, W).
            n_z: The number of slices Z.

        Returns:
            The decoder output before the activation, (B * Z, out_channels, H, W).
        """
        s4, s8, s16 = self._project(features)
        stem1 = self.stem1(image)
        stem2 = self.stem2(stem1)
        x = self._run(self.blocks16, self.lateral16(s16), n_z)
        lateral = self.lateral8(s8)
        x = self._run(self.blocks8, _resize_to(self.reduce8(x), lateral) + lateral, n_z)
        lateral = self.lateral4(s4)
        x = self._run(self.blocks4, _resize_to(self.reduce4(x), lateral) + lateral, n_z)
        x = self.fuse2(torch.cat([_resize_to(self.up2(x), stem2), stem2], dim=1))
        x = self.fuse1(torch.cat([_resize_to(self.up1(x), stem1), stem1], dim=1))
        return self.out_conv(x)


# The hybrid decoders by name. To add a decoder, implement the `HybridDecoder` interface and register it here.
# A decoder with deep supervision (e.g. CAGD-3D) can keep its auxiliary outputs of the last forward pass in an
# attribute, which `HybridUNETR.decode` would then unfold, activate and resize like the main output.
HYBRID_DECODERS = {"hiera_u": HieraUDecoder, "c3f2": C3F2Decoder}
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
        decoder: The decoder. Can either be a name from `HYBRID_DECODERS` ('hiera_u' or 'c3f2') or a torch module
            that implements the interface of `HybridDecoder` and has the attribute `in_channels`.
        decoder_kwargs: The keyword arguments for the decoder, if it is given by name.
        out_channels: The number of output channels. It is ignored if the decoder is a torch module.
        use_sam_stats: Whether to normalize the input data with the statistics of the pretrained SAM2 model.
        use_mae_stats: Whether to normalize the input data with the statistics of the pretrained MAE model.
        use_dino_stats: Whether to normalize the input data with the statistics of the pretrained DINO models.
        use_imagenet_stats: Whether to normalize with standard ImageNet statistics.
        resize_input: Whether to resize the input images to match `img_size`.
        encoder_checkpoint: Checkpoint for initializing the image encoder.
            Can either be a filepath or an already loaded checkpoint. If it is a full SAM2 checkpoint,
            the projections of the decoder are initialized from `conv_s0` and `conv_s1` of the SAM2 mask decoder.
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

        # Load the checkpoint only once, it is used for the encoder and for the projections of the decoder.
        checkpoint_state = None if encoder_checkpoint is None else _load_checkpoint_state(encoder_checkpoint)

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
            encoder_checkpoint=checkpoint_state,
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
        self._decoder_name = decoder if isinstance(decoder, str) else None
        self._decoder_kwargs = dict(decoder_kwargs or {})

        if checkpoint_state is not None and backbone == "sam2":
            self._init_skip_projections_from_state(checkpoint_state)

    def _init_skip_projections_from_state(self, state: Mapping) -> None:
        # Full SAM2 checkpoints contain the projections of the stride-4 and stride-8 features in the mask decoder.
        for name, prefix in (("proj_s4", "sam_mask_decoder.conv_s0."), ("proj_s8", "sam_mask_decoder.conv_s1.")):
            projection = getattr(self.decoder, name, None)
            if projection is None:
                continue
            projection_state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
            expected = {key: value.shape for key, value in projection.state_dict().items()}
            if {key: value.shape for key, value in projection_state.items()} == expected:
                projection.load_state_dict(projection_state)

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
        if self._decoder_name is None:
            raise RuntimeError("The decoder_config is only available for a decoder that is given by name.")
        return {
            "decoder": self._decoder_name, "decoder_kwargs": dict(self._decoder_kwargs),
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
        x, input_shape = self.preprocess(x)
        image = fold_z(x)
        _, levels = self._encode_image(image)
        if levels is None or len(levels) != 3:
            raise ValueError(
                "The encoder must return a tuple with the list of features at stride 4, 8 and 16 as second element."
            )
        return {"features": list(levels), "image": image, "n_z": n_z}, input_shape

    def decode(self, features, input_shape: Tuple[int, ...], original_shape: Tuple[int, ...]) -> torch.Tensor:
        """Run the decoder on the encoder features.

        Args:
            features: The dictionary returned by `encode`.
            input_shape: The spatial shape (Z, Y, X) after preprocessing, returned by `encode`.
            original_shape: The spatial shape of the original input, (Z, Y, X) or (Y, X).

        Returns:
            The output, resized to `original_shape`. It has no z axis if `original_shape` is (Y, X).
        """
        is_2d = len(original_shape) == 2
        if is_2d:
            original_shape = (1, *original_shape)

        n_z = features["n_z"]
        x = unfold_z(self.decoder(features["features"], features["image"], n_z), n_z)
        if self.final_activation is not None:
            x = self.final_activation(x)

        x = self.postprocess_masks(x, input_shape, original_shape)
        return x[:, :, 0] if is_2d else x
