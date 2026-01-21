from typing import (
    Any,
    Final,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
from typing_extensions import Unpack

#
# Model Internal Post-processing
#
# Note: these are mainly for bioimage.io models, where postprocessing has to be done
# inside of the model unless its defined in the general spec

AccumulatorMode = Literal["mean", "min", "max"]


class AccumulateChannels(nn.Module):
    """@private"""

    invariant_channels: Final[Optional[Tuple[int, int]]]
    accumulate_channels: Final[Tuple[int, int]]
    accumulator: Final[AccumulatorMode]

    def __init__(
        self,
        invariant_channels: Optional[Tuple[int, int]],
        accumulate_channels: Tuple[int, int],
        accumulator: AccumulatorMode,
    ):
        super().__init__()
        self.invariant_channels = invariant_channels
        self.accumulate_channels = accumulate_channels
        assert accumulator in ("mean", "min", "max")
        self.accumulator = accumulator

    def _accumulate(self, x: torch.Tensor, c0: int, c1: int):
        if self.accumulator == "mean":
            acc = torch.mean
        elif self.accumulator == "min":
            acc = torch.min
        elif self.accumulator == "max":
            acc = torch.max
        else:
            raise ValueError(f"Unknown accumulator: {self.accumulator}")

        res = acc(x[:, c0:c1], dim=1, keepdim=True)
        if not torch.is_tensor(res):
            res = res.values
        assert torch.is_tensor(res)
        return res

    def forward(self, x: torch.Tensor):
        if self.invariant_channels is None:
            c0, c1 = self.accumulate_channels
            return self._accumulate(x, c0, c1)
        else:
            i0, i1 = self.invariant_channels
            c0, c1 = self.accumulate_channels
            return torch.cat([x[:, i0:i1], self._accumulate(x, c0, c1)], dim=1)


def affinities_to_boundaries(
    aff_channels: Tuple[int, int], accumulator: AccumulatorMode = "max"
):
    """@private"""
    return AccumulateChannels(None, aff_channels, accumulator)


def affinities_with_foreground_to_boundaries(
    aff_channels: Tuple[int, int],
    fg_channel: Tuple[int, int] = (0, 1),
    accumulator: AccumulatorMode = "max",
):
    """@private"""
    return AccumulateChannels(fg_channel, aff_channels, accumulator)


def affinities_to_boundaries2d():
    """@private"""
    return affinities_to_boundaries((0, 2))


def affinities_with_foreground_to_boundaries2d():
    """@private"""
    return affinities_with_foreground_to_boundaries((1, 3))


def affinities_to_boundaries3d():
    """@private"""
    return affinities_to_boundaries((0, 3))


def affinities_with_foreground_to_boundaries3d():
    """@private"""
    return affinities_with_foreground_to_boundaries((1, 4))


def affinities_to_boundaries_anisotropic():
    """@private"""
    return AccumulateChannels(None, (1, 3), "max")


POSTPROCESSING = {
    "affinities_to_boundaries_anisotropic": affinities_to_boundaries_anisotropic,
    "affinities_to_boundaries2d": affinities_to_boundaries2d,
    "affinities_with_foreground_to_boundaries2d": affinities_with_foreground_to_boundaries2d,
    "affinities_to_boundaries3d": affinities_to_boundaries3d,
    "affinities_with_foreground_to_boundaries3d": affinities_with_foreground_to_boundaries3d,
}
"""@private
"""


#
# Base Implementations
#
class ConvBlock(nn.Module):
    """@private"""

    in_channels: Final[int]
    out_channels: Final[int]
    kernel_size: Final[int]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int,
        kernel_size: int = 3,
        padding: int = 1,
        norm: Optional[
            Literal["InstanceNorm", "InstanceNormTrackStats", "BatchNorm", "GroupNorm"]
        ] = "InstanceNorm",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        conv = nn.Conv2d if dim == 2 else nn.Conv3d

        if norm is None:
            self.block = nn.Sequential(
                conv(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=self.kernel_size,
                    padding=padding,
                ),
                nn.ReLU(inplace=True),
                conv(
                    self.out_channels,
                    self.out_channels,
                    kernel_size=self.kernel_size,
                    padding=padding,
                ),
                nn.ReLU(inplace=True),
            )
        else:
            self.block = nn.Sequential(
                get_norm_layer(norm, dim, self.in_channels),
                conv(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=self.kernel_size,
                    padding=padding,
                ),
                nn.ReLU(inplace=True),
                get_norm_layer(norm, dim, self.out_channels),
                conv(
                    self.out_channels,
                    self.out_channels,
                    kernel_size=self.kernel_size,
                    padding=padding,
                ),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor):
        return self.block(x)


class ConvBlock2d(ConvBlock):
    """@private"""

    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any):
        super().__init__(in_channels, out_channels, dim=2, **kwargs)


class ConvBlock3d(ConvBlock):
    """@private"""

    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any):
        super().__init__(in_channels, out_channels, dim=3, **kwargs)


InterpolationMode = Literal[
    "nearest", "linear", "bilinear", "bicubic", "trilinear", "area", "nearest-exact"
]


class Upsampler(nn.Module):
    """@private"""

    mode: Final[InterpolationMode]
    scale_factor: Final[Union[float, List[float]]]

    def __init__(
        self,
        scale_factor: Union[float, List[float]],
        in_channels: int,
        out_channels: int,
        dim: Literal[2, 3],
        mode: InterpolationMode,
    ):
        super().__init__()
        self.mode = mode
        self.scale_factor = (
            [float(sf) for sf in scale_factor]
            if isinstance(scale_factor, (list, tuple))
            else float(scale_factor)
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

        if dim == 2:
            self.conv = nn.Conv2d(self.in_channels, self.out_channels, 1)
        elif dim == 3:
            self.conv = nn.Conv3d(self.in_channels, self.out_channels, 1)
        else:
            raise NotImplementedError(f"Invalid dimension: {dim}")

    def forward(self, x: torch.Tensor):
        x = nn.functional.interpolate(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False
        )
        x = self.conv(x)
        return x


class Upsampler2d(Upsampler):
    """@private"""

    def __init__(
        self,
        scale_factor: Union[float, List[float]],
        in_channels: int,
        out_channels: int,
        mode: InterpolationMode = "bilinear",
    ):
        super().__init__(scale_factor, in_channels, out_channels, dim=2, mode=mode)


class Upsampler3d(Upsampler):
    """@private"""

    def __init__(
        self,
        scale_factor: Union[float, List[float]],
        in_channels: int,
        out_channels: int,
        mode: InterpolationMode = "trilinear",
    ):
        super().__init__(scale_factor, in_channels, out_channels, dim=3, mode=mode)


class ConvBlockKwargs(TypedDict, total=False):
    kernel_size: Tuple[int, ...]
    padding: Tuple[int, ...]


def _update_conv_kwargs(
    kwargs: ConvBlockKwargs, scale_factor: Union[float, Sequence[float]]
):
    # if the scale factor is a scalar or all entries are the same we don"t need to update the kwargs
    if isinstance(scale_factor, (int, float)) or len(set(scale_factor)) == 1:
        return kwargs
    else:  # otherwise set anisotropic kernel
        kernel_size = kwargs.get("kernel_size", 3)
        padding = kwargs.get("padding", 1)

        # bail out if kernel size or padding aren"t scalars, because it"s
        # unclear what to do in this case
        if not (isinstance(kernel_size, int) and isinstance(padding, int)):
            return kwargs

        kernel_size = tuple(
            1 if factor == 1 else kernel_size for factor in scale_factor
        )
        padding = tuple(0 if factor == 1 else padding for factor in scale_factor)
        kwargs.update({"kernel_size": kernel_size, "padding": padding})
        return kwargs


class Encoder(nn.Module):
    """@private"""

    in_channels: Final[int]
    out_channels: Final[int]
    depth: Final[int]

    def __init__(
        self,
        features: Sequence[int],
        scale_factors: Sequence[float],
        conv_block_impl: Type[Union[ConvBlock2d, ConvBlock3d]],
        pooler_impl: Type[nn.Module],
        anisotropic_kernel: bool = False,
        **conv_block_kwargs: Unpack[ConvBlockKwargs],
    ):
        super().__init__()
        if len(features) != len(scale_factors) + 1:
            raise ValueError(
                "Incompatible number of features {len(features)} and scale_factors {len(scale_factors)}"
            )

        conv_kwargs = [conv_block_kwargs] * len(scale_factors)
        if anisotropic_kernel:
            conv_kwargs = [
                _update_conv_kwargs(kwargs, scale_factor)
                for kwargs, scale_factor in zip(conv_kwargs, scale_factors)
            ]

        self.blocks = nn.ModuleList(
            [
                conv_block_impl(inc, outc, **kwargs)
                for inc, outc, kwargs in zip(features[:-1], features[1:], conv_kwargs)
            ]
        )
        self.depth = len(self.blocks)
        self.poolers = nn.ModuleList([pooler_impl(factor) for factor in scale_factors])

        self.in_channels = features[0]
        self.out_channels = features[-1]

    def __len__(self):
        return self.depth

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        encoder_out: List[torch.Tensor] = []
        for block, pooler in zip(self.blocks, self.poolers):
            x = block(x)
            encoder_out.append(x)
            x = pooler(x)

        return x, encoder_out


class Decoder(nn.Module):
    """@private"""

    in_channels: Final[int]
    out_channels: Final[int]
    depth: Final[int]

    def __init__(
        self,
        features: Sequence[int],
        scale_factors: Sequence[float],
        conv_block_impl: Type[Union[ConvBlock2d, ConvBlock3d]],
        sampler_impl: Type[nn.Module],
        anisotropic_kernel: bool = False,
        **conv_block_kwargs: Unpack[ConvBlockKwargs],
    ):
        super().__init__()
        self.in_channels = features[0]
        self.out_channels = features[-1]

        if len(features) != len(scale_factors) + 1:
            raise ValueError(
                "Incompatible number of features {len(features)} and scale_factors {len(scale_factors)}"
            )

        conv_kwargs = [conv_block_kwargs] * len(scale_factors)
        if anisotropic_kernel:
            conv_kwargs = [
                _update_conv_kwargs(kwargs, scale_factor)
                for kwargs, scale_factor in zip(conv_kwargs, scale_factors)
            ]

        self.blocks = nn.ModuleList(
            [
                conv_block_impl(inc, outc, **kwargs)
                for inc, outc, kwargs in zip(features[:-1], features[1:], conv_kwargs)
            ]
        )
        self.depth = len(self.blocks)
        self.samplers = nn.ModuleList(
            [
                sampler_impl(factor, inc, outc)
                for factor, inc, outc in zip(scale_factors, features[:-1], features[1:])
            ]
        )

    def __len__(self):
        return self.depth

    # FIXME this prevents traces from being valid for other input sizes, need to find
    # a solution to traceable cropping
    # note: maybe not as torch.narrow is scriptable...
    def _crop(self, x: torch.Tensor, shape: List[int]):
        shape_diff = [(xsh - sh) // 2 for xsh, sh in zip(x.shape, shape)]
        # Implementation with torch.narrow, does not fix the tracing warnings!
        # torch.narrow is scriptable though...
        for dim, (sh, sd) in enumerate(zip(shape, shape_diff)):
            x = torch.narrow(x, dim, sd, sh)
        return x

    def _concat(self, x1: torch.Tensor, x2: torch.Tensor):
        return torch.cat([x1, self._crop(x2, list(x1.shape))], dim=1)

    def forward(self, x: torch.Tensor, encoder_inputs: List[torch.Tensor]):
        if len(encoder_inputs) != len(self.blocks):
            raise ValueError(
                f"Invalid number of encoder_inputs: expect {len(self.blocks)}, got {len(encoder_inputs)}"
            )

        decoder_out: List[torch.Tensor] = []
        for i, (block, sampler) in enumerate(zip(self.blocks, self.samplers)):
            x = sampler(x)
            x = block(self._concat(x, encoder_inputs[i]))
            decoder_out.append(x)

        return decoder_out + [x]


class InstanceNormKwargs(TypedDict):
    affine: bool
    track_running_stats: bool
    momentum: float


def get_norm_layer(
    norm: Literal["InstanceNorm", "InstanceNormTrackStats", "BatchNorm", "GroupNorm"],
    dim: int,
    channels: int,
    n_groups: int = 32,
):
    """@private"""
    if norm == "InstanceNorm":
        return nn.InstanceNorm2d(channels) if dim == 2 else nn.InstanceNorm3d(channels)
    elif norm == "InstanceNormTrackStats":
        kwargs = InstanceNormKwargs(
            affine=True, track_running_stats=True, momentum=0.01
        )
        return (
            nn.InstanceNorm2d(channels, **kwargs)
            if dim == 2
            else nn.InstanceNorm3d(channels, **kwargs)
        )
    elif norm == "GroupNorm":
        return nn.GroupNorm(min(n_groups, channels), channels)
    elif norm == "BatchNorm":
        return nn.BatchNorm2d(channels) if dim == 2 else nn.BatchNorm3d(channels)
    else:
        raise ValueError(
            f"Invalid norm: expect one of 'InstanceNorm', 'BatchNorm' or 'GroupNorm', got {norm}"
        )


class UNetBase(nn.Module):
    """Base class for implementing a U-Net.

    Args:
        encoder: The encoder of the U-Net.
        base: The base layer of the U-Net.
        decoder: The decoder of the U-Net.
        out_conv: The output convolution applied after the last decoder layer.
        final_activation: The activation applied after the output convolution or last decoder layer.
        postprocessing: A postprocessing function to apply after the U-Net output.
        check_shape: Whether to check the input shape to the U-Net forward call.
    """

    return_decoder_outputs: Final[bool]
    check_shape: Final[bool]

    # encoder: Final[Encoder]
    # base: Final[nn.Module]
    # decoder: Final[Decoder]
    # out_conv: Final[Optional[Union[nn.Module, nn.ModuleList]]]
    # final_activation: Final[Optional[nn.Module]]
    # postprocessing: Final[Optional[nn.Module]]
    def __init__(
        self,
        encoder: Encoder,
        base: nn.Module,
        decoder: Decoder,
        out_conv: Optional[nn.Module] = None,
        final_activation: Optional[Union[nn.Module, str]] = None,
        postprocessing: Optional[Union[nn.Module, str]] = None,
        check_shape: bool = True,
    ):
        super().__init__()
        if len(encoder) != len(decoder):
            raise ValueError(
                f"Incompatible depth of encoder (depth={len(encoder)}) and decoder (depth={len(decoder)})"
            )

        self.encoder = encoder
        self.base = base
        self.decoder = decoder

        if out_conv is None:
            self.return_decoder_outputs = False
            self._out_channels = self.decoder.out_channels
        elif isinstance(out_conv, nn.ModuleList):
            if len(out_conv) != len(self.decoder):
                raise ValueError(
                    f"Invalid length of out_conv, expected {len(decoder)}, got {len(out_conv)}"
                )
            self.return_decoder_outputs = True
            self._out_channels = [conv.out_channels for conv in out_conv]
        else:
            self.return_decoder_outputs = False
            self._out_channels = out_conv.out_channels
        self.out_conv = out_conv
        self.check_shape = check_shape
        self.final_activation = self._get_activation(final_activation)
        self.postprocessing = self._get_postprocessing(postprocessing)

    @property
    def in_channels(self):
        return self.encoder.in_channels

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def depth(self):
        return self.encoder.depth

    def _get_activation(self, activation: Optional[Union[str, nn.Module]]):
        if activation is None or isinstance(activation, nn.Module):
            return activation
        elif not isinstance(activation, str):
            raise TypeError(f"Invalid activation type: {type(activation)}")

        activation_class = getattr(nn, activation, None)
        if activation_class is None:
            raise ValueError(f"Invalid activation: {activation}")

        ret = activation_class()
        if not isinstance(ret, nn.Module):
            raise ValueError(f"Invalid activation: {activation}")

        return ret

    def _get_postprocessing(self, postprocessing: Optional[Union[str, nn.Module]]):
        if postprocessing is None or isinstance(postprocessing, nn.Module):
            return postprocessing
        elif postprocessing in POSTPROCESSING:
            return POSTPROCESSING[postprocessing]()
        else:
            raise ValueError(f"Invalid postprocessing: {postprocessing}")

    # load encoder / decoder / base states for pretraining
    def load_encoder_state(self, state: Mapping[str, Any]):
        _ = self.encoder.load_state_dict(state)

    def load_decoder_state(self, state: Mapping[str, Any]):
        _ = self.decoder.load_state_dict(state)

    def load_base_state(self, state: Mapping[str, Any]):
        _ = self.base.load_state_dict(state)

    def _apply_default(self, x: torch.Tensor) -> torch.Tensor:
        x, encoder_out = self.encoder(x)
        x = self.base(x)
        x = self.decoder(x, encoder_inputs=encoder_out[::-1])[-1]

        if self.out_conv is not None:
            x = self.out_conv(x)
        if self.final_activation is not None:
            x = self.final_activation(x)
        if self.postprocessing is not None:
            x = self.postprocessing(x)

        return x

    def _apply_with_side_outputs(self, x: torch.Tensor):
        x, encoder_out = self.encoder(x)
        x = self.base(x)
        ret = self.decoder(x, encoder_inputs=encoder_out[::-1])

        assert isinstance(self.out_conv, nn.ModuleList)
        ret = [conv(xx) for xx, conv in zip(ret, self.out_conv)]
        if self.final_activation is not None:
            ret = [self.final_activation(xx) for xx in ret]

        if self.postprocessing is not None:
            ret = [self.postprocessing(xx) for xx in ret]

        # we reverse the list to have the full shape output as first element
        return ret[::-1]

    @torch.jit.ignore
    def _check_shape(self, x: torch.Tensor):
        spatial_shape = tuple(x.shape)[2:]
        depth = len(self.encoder)
        factor = [2**depth] * len(spatial_shape)
        if any(sh % fac != 0 for sh, fac in zip(spatial_shape, factor)):
            msg = (
                f"Invalid shape for U-Net: {spatial_shape} is not divisible by {factor}"
            )
            raise ValueError(msg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply U-Net to input data.

        Args:
            x: The input data.

        Returns:
            The output of the U-Net.
        """
        # Cast input data to float, hotfix for modelzoo deployment issues, leaving it here for reference.
        # x = x.float()
        if self.check_shape:
            self._check_shape(x)

        if self.return_decoder_outputs:
            return self._apply_with_side_outputs(x)
        else:
            return self._apply_default(x)


#
# 2d unet implementations
#


class UNet2d(UNetBase):
    """A 2D U-Net network for segmentation and other image-to-image tasks.

    The number of features for each level of the U-Net are computed as follows: initial_features * gain ** level.
    The number of levels is determined by the depth argument. By default the U-Net uses two convolutional layers
    per level, max-pooling for downsampling and linear interpolation for upsampling.
    These implementations can be changed by providing arguments for `conv_block_impl`, `pooler_impl`
    and `sampler_impl` respectively.

    Args:
        in_channels: The number of input image channels.
        out_channels: The number of output image channels.
        depth: The number of encoder / decoder levels of the U-Net.
        initial_features: The initial number of features, corresponding to the features of the first conv block.
        gain: The gain factor for increasing the features after each level.
        final_activation: The activation applied after the output convolution or last decoder layer.
        return_side_outputs: Whether to return the outputs after each decoder level.
        conv_block_impl: The implementation of the convolutional block.
        pooler_impl: The implementation of the pooling layer.
        postprocessing: A postprocessing function to apply after the U-Net output.
        check_shape: Whether to check the input shape to the U-Net forward call.
        conv_block_kwargs: The keyword arguments for the convolutional block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Union[Optional[int], Sequence[Optional[int]]],
        depth: int = 4,
        initial_features: int = 32,
        gain: int = 2,
        final_activation: Optional[str] = None,
        return_side_outputs: bool = False,
        conv_block_impl: Type[Union[ConvBlock2d, ConvBlock3d]] = ConvBlock2d,
        pooler_impl: Type[nn.Module] = nn.MaxPool2d,
        sampler_impl: Type[nn.Module] = Upsampler2d,
        postprocessing: Optional[Union[nn.Module, str]] = None,
        check_shape: bool = True,
        **conv_block_kwargs: Unpack[ConvBlockKwargs],
    ):
        features_encoder = [in_channels] + [
            initial_features * gain**i for i in range(depth)
        ]
        features_decoder: List[int] = [
            initial_features * gain**i for i in range(depth + 1)
        ][::-1]
        scale_factors = depth * (2,)

        if return_side_outputs:
            if isinstance(out_channels, int) or out_channels is None:
                out_channels = [out_channels] * depth
            if len(out_channels) != depth:
                raise ValueError()
            out_conv = nn.ModuleList(
                [
                    nn.Conv2d(feat, outc, 1)
                    for feat, outc in zip(features_decoder[1:], out_channels)
                ]
            )
        else:
            out_conv = (
                None
                if out_channels is None
                else nn.Conv2d(features_decoder[-1], out_channels, 1)
            )

        super().__init__(
            encoder=Encoder(
                features=features_encoder,
                scale_factors=scale_factors,
                conv_block_impl=conv_block_impl,
                pooler_impl=pooler_impl,
                **conv_block_kwargs,
            ),
            decoder=Decoder(
                features=features_decoder,
                scale_factors=scale_factors[::-1],
                conv_block_impl=conv_block_impl,
                sampler_impl=sampler_impl,
                **conv_block_kwargs,
            ),
            base=conv_block_impl(
                features_encoder[-1], features_encoder[-1] * gain, **conv_block_kwargs
            ),
            out_conv=out_conv,
            final_activation=final_activation,
            postprocessing=postprocessing,
            check_shape=check_shape,
        )
        self.init_kwargs = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "depth": depth,
            "initial_features": initial_features,
            "gain": gain,
            "final_activation": final_activation,
            "return_side_outputs": return_side_outputs,
            "conv_block_impl": conv_block_impl,
            "pooler_impl": pooler_impl,
            "sampler_impl": sampler_impl,
            "postprocessing": postprocessing,
            **conv_block_kwargs,
        }


#
# 3d unet implementations
#


class AnisotropicUNet(UNetBase):
    """A 3D U-Net network for segmentation and other image-to-image tasks.

    The number of features for each level of the U-Net are computed as follows: initial_features * gain ** level.
    The number of levels is determined by the length of the scale_factors argument.
    The scale factors determine the pooling factors for each level. By specifying [1, 2, 2] the pooling
    is done in an anisotropic fashion, i.e. only across the xy-plane,
    by specifying [2, 2, 2] it is done in an isotropic fashion.

    By default the U-Net uses two convolutional layers per level.
    This can be changed by providing an argument for `conv_block_impl`.

    Args:
        in_channels: The number of input image channels.
        out_channels: The number of output image channels.
        scale_factors: The factors for max pooling for the levels of the U-Net.
        initial_features: The initial number of features, corresponding to the features of the first conv block.
        gain: The gain factor for increasing the features after each level.
        final_activation: The activation applied after the output convolution or last decoder layer.
        return_side_outputs: Whether to return the outputs after each decoder level.
        conv_block_impl: The implementation of the convolutional block.
        anisotropic_kernel: Whether to use an anisotropic kernel in addition to anisotropic scaling factor.
        postprocessing: A postprocessing function to apply after the U-Net output.
        check_shape: Whether to check the input shape to the U-Net forward call.
        conv_block_kwargs: The keyword arguments for the convolutional block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factors: Sequence[float],
        initial_features: int = 32,
        gain: int = 2,
        final_activation: Optional[Union[str, nn.Module]] = None,
        return_side_outputs: bool = False,
        conv_block_impl: Type[nn.Module] = ConvBlock3d,
        anisotropic_kernel: bool = False,
        postprocessing: Optional[Union[str, nn.Module]] = None,
        check_shape: bool = True,
        **conv_block_kwargs: Unpack[ConvBlockKwargs],
    ):
        depth = len(scale_factors)
        features_encoder = [in_channels] + [
            initial_features * gain**i for i in range(depth)
        ]
        features_decoder = [initial_features * gain**i for i in range(depth + 1)][::-1]

        if return_side_outputs:
            if isinstance(out_channels, int) or out_channels is None:
                out_channels = [out_channels] * depth
            if len(out_channels) != depth:
                raise ValueError()
            out_conv = nn.ModuleList(
                [
                    nn.Conv3d(feat, outc, 1)
                    for feat, outc in zip(features_decoder[1:], out_channels)
                ]
            )
        else:
            out_conv = (
                None
                if out_channels is None
                else nn.Conv3d(features_decoder[-1], out_channels, 1)
            )

        super().__init__(
            encoder=Encoder(
                features=features_encoder,
                scale_factors=scale_factors,
                conv_block_impl=conv_block_impl,
                pooler_impl=nn.MaxPool3d,
                anisotropic_kernel=anisotropic_kernel,
                **conv_block_kwargs,
            ),
            decoder=Decoder(
                features=features_decoder,
                scale_factors=scale_factors[::-1],
                conv_block_impl=conv_block_impl,
                sampler_impl=Upsampler3d,
                anisotropic_kernel=anisotropic_kernel,
                **conv_block_kwargs,
            ),
            base=conv_block_impl(
                features_encoder[-1], features_encoder[-1] * gain, **conv_block_kwargs
            ),
            out_conv=out_conv,
            final_activation=final_activation,
            postprocessing=postprocessing,
            check_shape=check_shape,
        )
        self.init_kwargs = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "scale_factors": scale_factors,
            "initial_features": initial_features,
            "gain": gain,
            "final_activation": final_activation,
            "return_side_outputs": return_side_outputs,
            "conv_block_impl": conv_block_impl,
            "anisotropic_kernel": anisotropic_kernel,
            "postprocessing": postprocessing,
            **conv_block_kwargs,
        }

    def _check_shape(self, x: torch.Tensor):
        spatial_shape = tuple(x.shape)[2:]
        scale_factors = self.init_kwargs.get(
            "scale_factors", [[2, 2, 2]] * len(self.encoder)
        )
        factor = [int(np.prod([sf[i] for sf in scale_factors])) for i in range(3)]
        if len(spatial_shape) != len(factor):
            msg = f"Invalid shape for U-Net: dimensions don't agree {len(spatial_shape)} != {len(factor)}"
            raise ValueError(msg)
        if any(sh % fac != 0 for sh, fac in zip(spatial_shape, factor)):
            msg = (
                f"Invalid shape for U-Net: {spatial_shape} is not divisible by {factor}"
            )
            raise ValueError(msg)


class UNet3d(AnisotropicUNet):
    """A 3D U-Net network for segmentation and other image-to-image tasks.

    This class uses the same implementation as `AnisotropicUNet`, with isotropic scaling in each level.

    Args:
        in_channels: The number of input image channels.
        out_channels: The number of output image channels.
        depth: The number of encoder / decoder levels of the U-Net.
        initial_features: The initial number of features, corresponding to the features of the first conv block.
        gain: The gain factor for increasing the features after each level.
        final_activation: The activation applied after the output convolution or last decoder layer.
        return_side_outputs: Whether to return the outputs after each decoder level.
        conv_block_impl: The implementation of the convolutional block.
        postprocessing: A postprocessing function to apply after the U-Net output.
        check_shape: Whether to check the input shape to the U-Net forward call.
        conv_block_kwargs: The keyword arguments for the convolutional block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 4,
        initial_features: int = 32,
        gain: int = 2,
        final_activation: Optional[Union[str, nn.Module]] = None,
        return_side_outputs: bool = False,
        conv_block_impl: Type[nn.Module] = ConvBlock3d,
        postprocessing: Optional[Union[str, nn.Module]] = None,
        check_shape: bool = True,
        **conv_block_kwargs: Unpack[ConvBlockKwargs],
    ):
        scale_factors = depth * [2]
        super().__init__(
            in_channels,
            out_channels,
            scale_factors,
            initial_features=initial_features,
            gain=gain,
            final_activation=final_activation,
            return_side_outputs=return_side_outputs,
            anisotropic_kernel=False,
            postprocessing=postprocessing,
            conv_block_impl=conv_block_impl,
            check_shape=check_shape,
            **conv_block_kwargs,
        )
        self.init_kwargs = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "depth": depth,
            "initial_features": initial_features,
            "gain": gain,
            "final_activation": final_activation,
            "return_side_outputs": return_side_outputs,
            "conv_block_impl": conv_block_impl,
            "postprocessing": postprocessing,
            **conv_block_kwargs,
        }
