from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn


#
# Model Internal Post-processing
#
# Note: these are mainly for bioimage.io models, where postprocessing has to be done
# inside of the model unless its defined in the general spec


class AccumulateChannels(nn.Module):
    """@private
    """
    def __init__(
        self,
        invariant_channels,
        accumulate_channels,
        accumulator
    ):
        super().__init__()
        self.invariant_channels = invariant_channels
        self.accumulate_channels = accumulate_channels
        assert accumulator in ("mean", "min", "max")
        self.accumulator = getattr(torch, accumulator)

    def _accumulate(self, x, c0, c1):
        res = self.accumulator(x[:, c0:c1], dim=1, keepdim=True)
        if not torch.is_tensor(res):
            res = res.values
        assert torch.is_tensor(res)
        return res

    def forward(self, x):
        if self.invariant_channels is None:
            c0, c1 = self.accumulate_channels
            return self._accumulate(x, c0, c1)
        else:
            i0, i1 = self.invariant_channels
            c0, c1 = self.accumulate_channels
            return torch.cat([x[:, i0:i1], self._accumulate(x, c0, c1)], dim=1)


def affinities_to_boundaries(aff_channels, accumulator="max"):
    """@private
    """
    return AccumulateChannels(None, aff_channels, accumulator)


def affinities_with_foreground_to_boundaries(aff_channels, fg_channel=(0, 1), accumulator="max"):
    """@private
    """
    return AccumulateChannels(fg_channel, aff_channels, accumulator)


def affinities_to_boundaries2d():
    """@private
    """
    return affinities_to_boundaries((0, 2))


def affinities_with_foreground_to_boundaries2d():
    """@private
    """
    return affinities_with_foreground_to_boundaries((1, 3))


def affinities_to_boundaries3d():
    """@private
    """
    return affinities_to_boundaries((0, 3))


def affinities_with_foreground_to_boundaries3d():
    """@private
    """
    return affinities_with_foreground_to_boundaries((1, 4))


def affinities_to_boundaries_anisotropic():
    """@private
    """
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
    def __init__(
        self,
        encoder: nn.Module,
        base: nn.Module,
        decoder: nn.Module,
        out_conv: Optional[nn.Module] = None,
        final_activation: Optional[Union[nn.Module, str]] = None,
        postprocessing: Optional[Union[nn.Module, str]] = None,
        check_shape: bool = True,
    ):
        super().__init__()
        if len(encoder) != len(decoder):
            raise ValueError(f"Incompatible depth of encoder (depth={len(encoder)}) and decoder (depth={len(decoder)})")

        self.encoder = encoder
        self.base = base
        self.decoder = decoder

        if out_conv is None:
            self.return_decoder_outputs = False
            self._out_channels = self.decoder.out_channels
        elif isinstance(out_conv, nn.ModuleList):
            if len(out_conv) != len(self.decoder):
                raise ValueError(f"Invalid length of out_conv, expected {len(decoder)}, got {len(out_conv)}")
            self.return_decoder_outputs = True
            self._out_channels = [None if conv is None else conv.out_channels for conv in out_conv]
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
        return len(self.encoder)

    def _get_activation(self, activation):
        return_activation = None
        if activation is None:
            return None
        if isinstance(activation, nn.Module):
            return activation
        if isinstance(activation, str):
            return_activation = getattr(nn, activation, None)
        if return_activation is None:
            raise ValueError(f"Invalid activation: {activation}")
        return return_activation()

    def _get_postprocessing(self, postprocessing):
        if postprocessing is None:
            return None
        elif isinstance(postprocessing, nn.Module):
            return postprocessing
        elif postprocessing in POSTPROCESSING:
            return POSTPROCESSING[postprocessing]()
        else:
            raise ValueError(f"Invalid postprocessing: {postprocessing}")

    # load encoder / decoder / base states for pretraining
    def load_encoder_state(self, state):
        self.encoder.load_state_dict(state)

    def load_decoder_state(self, state):
        self.decoder.load_state_dict(state)

    def load_base_state(self, state):
        self.base.load_state_dict(state)

    def _apply_default(self, x):
        self.encoder.return_outputs = True
        self.decoder.return_outputs = False

        x, encoder_out = self.encoder(x)
        x = self.base(x)
        x = self.decoder(x, encoder_inputs=encoder_out[::-1])

        if self.out_conv is not None:
            x = self.out_conv(x)
        if self.final_activation is not None:
            x = self.final_activation(x)
        if self.postprocessing is not None:
            x = self.postprocessing(x)

        return x

    def _apply_with_side_outputs(self, x):
        self.encoder.return_outputs = True
        self.decoder.return_outputs = True

        x, encoder_out = self.encoder(x)
        x = self.base(x)
        x = self.decoder(x, encoder_inputs=encoder_out[::-1])

        x = [x if conv is None else conv(xx) for xx, conv in zip(x, self.out_conv)]
        if self.final_activation is not None:
            x = [self.final_activation(xx) for xx in x]

        if self.postprocessing is not None:
            x = [self.postprocessing(xx) for xx in x]

        # we reverse the list to have the full shape output as first element
        return x[::-1]

    def _check_shape(self, x):
        spatial_shape = tuple(x.shape)[2:]
        depth = len(self.encoder)
        factor = [2**depth] * len(spatial_shape)
        if any(sh % fac != 0 for sh, fac in zip(spatial_shape, factor)):
            msg = f"Invalid shape for U-Net: {spatial_shape} is not divisible by {factor}"
            raise ValueError(msg)

    def forward(self, x: torch.Tensor) -> torch.tensor:
        """Apply U-Net to input data.

        Args:
            x: The input data.

        Returns:
            The output of the U-Net.
        """
        # Cast input data to float, hotfix for modelzoo deployment issues, leaving it here for reference.
        # x = x.float()
        if getattr(self, "check_shape", True):
            self._check_shape(x)
        if self.return_decoder_outputs:
            return self._apply_with_side_outputs(x)
        else:
            return self._apply_default(x)


def _update_conv_kwargs(kwargs, scale_factor):
    # if the scale factor is a scalar or all entries are the same we don"t need to update the kwargs
    if isinstance(scale_factor, int) or scale_factor.count(scale_factor[0]) == len(scale_factor):
        return kwargs
    else:  # otherwise set anisotropic kernel
        kernel_size = kwargs.get("kernel_size", 3)
        padding = kwargs.get("padding", 1)

        # bail out if kernel size or padding aren"t scalars, because it"s
        # unclear what to do in this case
        if not (isinstance(kernel_size, int) and isinstance(padding, int)):
            return kwargs

        kernel_size = tuple(1 if factor == 1 else kernel_size for factor in scale_factor)
        padding = tuple(0 if factor == 1 else padding for factor in scale_factor)
        kwargs.update({"kernel_size": kernel_size, "padding": padding})
        return kwargs


class Encoder(nn.Module):
    """@private
    """
    def __init__(
        self,
        features,
        scale_factors,
        conv_block_impl,
        pooler_impl,
        anisotropic_kernel=False,
        **conv_block_kwargs
    ):
        super().__init__()
        if len(features) != len(scale_factors) + 1:
            raise ValueError("Incompatible number of features {len(features)} and scale_factors {len(scale_factors)}")

        conv_kwargs = [conv_block_kwargs] * len(scale_factors)
        if anisotropic_kernel:
            conv_kwargs = [_update_conv_kwargs(kwargs, scale_factor)
                           for kwargs, scale_factor in zip(conv_kwargs, scale_factors)]

        self.blocks = nn.ModuleList(
            [conv_block_impl(inc, outc, **kwargs)
             for inc, outc, kwargs in zip(features[:-1], features[1:], conv_kwargs)]
        )
        self.poolers = nn.ModuleList(
            [pooler_impl(factor) for factor in scale_factors]
        )
        self.return_outputs = True

        self.in_channels = features[0]
        self.out_channels = features[-1]

    def __len__(self):
        return len(self.blocks)

    def forward(self, x):
        encoder_out = []
        for block, pooler in zip(self.blocks, self.poolers):
            x = block(x)
            encoder_out.append(x)
            x = pooler(x)

        if self.return_outputs:
            return x, encoder_out
        else:
            return x


class Decoder(nn.Module):
    """@private
    """
    def __init__(
        self,
        features,
        scale_factors,
        conv_block_impl,
        sampler_impl,
        anisotropic_kernel=False,
        **conv_block_kwargs
    ):
        super().__init__()
        if len(features) != len(scale_factors) + 1:
            raise ValueError("Incompatible number of features {len(features)} and scale_factors {len(scale_factors)}")

        conv_kwargs = [conv_block_kwargs] * len(scale_factors)
        if anisotropic_kernel:
            conv_kwargs = [_update_conv_kwargs(kwargs, scale_factor)
                           for kwargs, scale_factor in zip(conv_kwargs, scale_factors)]

        self.blocks = nn.ModuleList(
            [conv_block_impl(inc, outc, **kwargs)
             for inc, outc, kwargs in zip(features[:-1], features[1:], conv_kwargs)]
        )
        self.samplers = nn.ModuleList(
            [sampler_impl(factor, inc, outc) for factor, inc, outc
             in zip(scale_factors, features[:-1], features[1:])]
        )
        self.return_outputs = False

        self.in_channels = features[0]
        self.out_channels = features[-1]

    def __len__(self):
        return len(self.blocks)

    # FIXME this prevents traces from being valid for other input sizes, need to find
    # a solution to traceable cropping
    def _crop(self, x, shape):
        shape_diff = [(xsh - sh) // 2 for xsh, sh in zip(x.shape, shape)]
        crop = tuple([slice(sd, xsh - sd) for sd, xsh in zip(shape_diff, x.shape)])
        return x[crop]
        # # Implementation with torch.narrow, does not fix the tracing warnings!
        # for dim, (sh, sd) in enumerate(zip(shape, shape_diff)):
        #     x = torch.narrow(x, dim, sd, sh)
        # return x

    def _concat(self, x1, x2):
        return torch.cat([x1, self._crop(x2, x1.shape)], dim=1)

    def forward(self, x, encoder_inputs):
        if len(encoder_inputs) != len(self.blocks):
            raise ValueError(f"Invalid number of encoder_inputs: expect {len(self.blocks)}, got {len(encoder_inputs)}")

        decoder_out = []
        for block, sampler, from_encoder in zip(self.blocks, self.samplers, encoder_inputs):
            x = sampler(x)
            x = block(self._concat(x, from_encoder))
            decoder_out.append(x)

        if self.return_outputs:
            return decoder_out + [x]
        else:
            return x


def get_norm_layer(norm, dim, channels, n_groups=32):
    """@private
    """
    if norm is None:
        return None
    if norm == "InstanceNorm":
        return nn.InstanceNorm2d(channels) if dim == 2 else nn.InstanceNorm3d(channels)
    elif norm == "InstanceNormTrackStats":
        kwargs = {"affine": True, "track_running_stats": True, "momentum": 0.01}
        return nn.InstanceNorm2d(channels, **kwargs) if dim == 2 else nn.InstanceNorm3d(channels, **kwargs)
    elif norm == "GroupNorm":
        return nn.GroupNorm(min(n_groups, channels), channels)
    elif norm == "BatchNorm":
        return nn.BatchNorm2d(channels) if dim == 2 else nn.BatchNorm3d(channels)
    else:
        raise ValueError(f"Invalid norm: expect one of 'InstanceNorm', 'BatchNorm' or 'GroupNorm', got {norm}")


class ConvBlock(nn.Module):
    """@private
    """
    def __init__(self, in_channels, out_channels, dim, kernel_size=3, padding=1, norm="InstanceNorm"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        conv = nn.Conv2d if dim == 2 else nn.Conv3d

        if norm is None:
            self.block = nn.Sequential(
                conv(in_channels, out_channels,
                     kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True),
                conv(out_channels, out_channels,
                     kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                get_norm_layer(norm, dim, in_channels),
                conv(in_channels, out_channels,
                     kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True),
                get_norm_layer(norm, dim, out_channels),
                conv(out_channels, out_channels,
                     kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.block(x)


class Upsampler(nn.Module):
    """@private
    """
    def __init__(self, scale_factor, in_channels, out_channels, dim, mode):
        super().__init__()
        self.mode = mode
        self.scale_factor = scale_factor

        conv = nn.Conv2d if dim == 2 else nn.Conv3d
        self.conv = conv(in_channels, out_channels, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        x = self.conv(x)
        return x


#
# 2d unet implementations
#

class ConvBlock2d(ConvBlock):
    """@private
    """
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, dim=2, **kwargs)


class Upsampler2d(Upsampler):
    """@private
    """
    def __init__(self, scale_factor,
                 in_channels, out_channels,
                 mode="bilinear"):
        super().__init__(scale_factor, in_channels, out_channels, dim=2, mode=mode)


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
        out_channels: int,
        depth: int = 4,
        initial_features: int = 32,
        gain: int = 2,
        final_activation=None,
        return_side_outputs: bool = False,
        conv_block_impl: nn.Module = ConvBlock2d,
        pooler_impl: nn.Module = nn.MaxPool2d,
        sampler_impl: nn.Module = Upsampler2d,
        postprocessing: Optional[Union[nn.Module, str]] = None,
        check_shape: bool = True,
        **conv_block_kwargs,
    ):
        features_encoder = [in_channels] + [initial_features * gain ** i for i in range(depth)]
        features_decoder = [initial_features * gain ** i for i in range(depth + 1)][::-1]
        scale_factors = depth * [2]

        if return_side_outputs:
            if isinstance(out_channels, int) or out_channels is None:
                out_channels = [out_channels] * depth
            if len(out_channels) != depth:
                raise ValueError()
            out_conv = nn.ModuleList(
                [nn.Conv2d(feat, outc, 1) for feat, outc in zip(features_decoder[1:], out_channels)]
            )
        else:
            out_conv = None if out_channels is None else nn.Conv2d(features_decoder[-1], out_channels, 1)

        super().__init__(
            encoder=Encoder(
                features=features_encoder,
                scale_factors=scale_factors,
                conv_block_impl=conv_block_impl,
                pooler_impl=pooler_impl,
                **conv_block_kwargs
            ),
            decoder=Decoder(
                features=features_decoder,
                scale_factors=scale_factors[::-1],
                conv_block_impl=conv_block_impl,
                sampler_impl=sampler_impl,
                **conv_block_kwargs
            ),
            base=conv_block_impl(
                features_encoder[-1], features_encoder[-1] * gain,
                **conv_block_kwargs
            ),
            out_conv=out_conv,
            final_activation=final_activation,
            postprocessing=postprocessing,
            check_shape=check_shape,
        )
        self.init_kwargs = {"in_channels": in_channels, "out_channels": out_channels, "depth": depth,
                            "initial_features": initial_features, "gain": gain,
                            "final_activation": final_activation, "return_side_outputs": return_side_outputs,
                            "conv_block_impl": conv_block_impl, "pooler_impl": pooler_impl,
                            "sampler_impl": sampler_impl, "postprocessing": postprocessing, **conv_block_kwargs}


#
# 3d unet implementations
#

class ConvBlock3d(ConvBlock):
    """@private
    """
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, dim=3, **kwargs)


class Upsampler3d(Upsampler):
    """@private
    """
    def __init__(self, scale_factor, in_channels, out_channels, mode="trilinear"):
        super().__init__(scale_factor, in_channels, out_channels, dim=3, mode=mode)


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
        scale_factors: List[List[int]],
        initial_features: int = 32,
        gain: int = 2,
        final_activation: Optional[Union[str, nn.Module]] = None,
        return_side_outputs: bool = False,
        conv_block_impl: nn.Module = ConvBlock3d,
        anisotropic_kernel: bool = False,
        postprocessing: Optional[Union[str, nn.Module]] = None,
        check_shape: bool = True,
        **conv_block_kwargs,
    ):
        depth = len(scale_factors)
        features_encoder = [in_channels] + [initial_features * gain ** i for i in range(depth)]
        features_decoder = [initial_features * gain ** i for i in range(depth + 1)][::-1]

        if return_side_outputs:
            if isinstance(out_channels, int) or out_channels is None:
                out_channels = [out_channels] * depth
            if len(out_channels) != depth:
                raise ValueError()
            out_conv = nn.ModuleList(
                [nn.Conv3d(feat, outc, 1) for feat, outc in zip(features_decoder[1:], out_channels)]
            )
        else:
            out_conv = None if out_channels is None else nn.Conv3d(features_decoder[-1], out_channels, 1)

        super().__init__(
            encoder=Encoder(
                features=features_encoder,
                scale_factors=scale_factors,
                conv_block_impl=conv_block_impl,
                pooler_impl=nn.MaxPool3d,
                anisotropic_kernel=anisotropic_kernel,
                **conv_block_kwargs
            ),
            decoder=Decoder(
                features=features_decoder,
                scale_factors=scale_factors[::-1],
                conv_block_impl=conv_block_impl,
                sampler_impl=Upsampler3d,
                anisotropic_kernel=anisotropic_kernel,
                **conv_block_kwargs
            ),
            base=conv_block_impl(
                features_encoder[-1], features_encoder[-1] * gain, **conv_block_kwargs
            ),
            out_conv=out_conv,
            final_activation=final_activation,
            postprocessing=postprocessing,
            check_shape=check_shape,
        )
        self.init_kwargs = {"in_channels": in_channels, "out_channels": out_channels, "scale_factors": scale_factors,
                            "initial_features": initial_features, "gain": gain,
                            "final_activation": final_activation, "return_side_outputs": return_side_outputs,
                            "conv_block_impl": conv_block_impl, "anisotropic_kernel": anisotropic_kernel,
                            "postprocessing": postprocessing, **conv_block_kwargs}

    def _check_shape(self, x):
        spatial_shape = tuple(x.shape)[2:]
        scale_factors = self.init_kwargs.get("scale_factors", [[2, 2, 2]]*len(self.encoder))
        factor = [int(np.prod([sf[i] for sf in scale_factors])) for i in range(3)]
        if len(spatial_shape) != len(factor):
            msg = f"Invalid shape for U-Net: dimensions don't agree {len(spatial_shape)} != {len(factor)}"
            raise ValueError(msg)
        if any(sh % fac != 0 for sh, fac in zip(spatial_shape, factor)):
            msg = f"Invalid shape for U-Net: {spatial_shape} is not divisible by {factor}"
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
        conv_block_impl: nn.Module = ConvBlock3d,
        postprocessing: Optional[Union[str, nn.Module]] = None,
        check_shape: bool = True,
        **conv_block_kwargs,
    ):
        scale_factors = depth * [2]
        super().__init__(in_channels, out_channels, scale_factors,
                         initial_features=initial_features, gain=gain,
                         final_activation=final_activation,
                         return_side_outputs=return_side_outputs,
                         anisotropic_kernel=False,
                         postprocessing=postprocessing,
                         conv_block_impl=conv_block_impl,
                         check_shape=check_shape,
                         **conv_block_kwargs)
        self.init_kwargs = {"in_channels": in_channels, "out_channels": out_channels, "depth": depth,
                            "initial_features": initial_features, "gain": gain,
                            "final_activation": final_activation, "return_side_outputs": return_side_outputs,
                            "conv_block_impl": conv_block_impl, "postprocessing": postprocessing, **conv_block_kwargs}
