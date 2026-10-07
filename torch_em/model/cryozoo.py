"""CryoZoo: pretrained models for cryo-electron tomography (cryo-ET) in the torch-em segmentation framework.

CryoZoo integrates published pretrained cryo-ET models into torch-em, so that you can finetune and use them
for segmentation like the other torch-em models.
CryoSiam is one of these integrations, and the CryoZooUNet uses its pretrained encoder.

CryoSiam is described in https://doi.org/10.1101/2025.11.11.687379.
Please cite it if you use this piece of code or pretrained weights in your research.

The authors release the weights under the MIT license at https://huggingface.co/frosinastojanovska/cryosiam_v1.0.
This module does not contain code from CryoSiam. It builds the encoder from the ResNet of MONAI, like CryoSiam does.
Thus, the weights of the CryoSiam checkpoint load without changes.
"""

import os
from collections import OrderedDict
from typing import List, Optional, Union

import numpy as np

import torch
import torch.nn as nn

try:
    from monai.networks.nets import ResNet
except ImportError:
    ResNet = None

from ..util.util import get_cache_directory
from .unet import UNetBase, Decoder, ConvBlock3d, Upsampler3d


CRYOSIAM_URL = "https://huggingface.co/frosinastojanovska/cryosiam_v1.0/resolve/main/dense_simsiam_pretrained.ckpt"
CRYOSIAM_CHECKSUM = "19957ef7eba45658432a81d37ac1764db65f97ddbfcd55bee3c190f51a9b17e7"

ENCODERS = {}  # The encoders of the CryoZooUNet by name. The register_encoder decorator fills it at import.


def register_encoder(name: str):
    """Add an encoder class to `ENCODERS` under the given name.

    Use this function as a decorator for the encoder class.

    Args:
        name: The name of the encoder.

    Returns:
        The decorator, which adds the encoder class and returns it unchanged.
    """
    def register(encoder_class):
        ENCODERS[name] = encoder_class
        return encoder_class
    return register


class CryoZooEncoder(nn.Module):
    """The base class for the CNN encoders of the CryoZooUNet.

    To add an encoder, make a subclass and decorate it with `register_encoder`. The subclass must:
    - set `features`: the number of channels of each skip connection and of the bottleneck, from fine to coarse.
    - set `scale_factors`: the downsampling factor between two consecutive entries of `features`.
    - set `stem_scale_factor`: the downsampling factor between the input and the first skip connection.
    - implement `forward`: it returns the bottleneck and the list of skip connections, from fine to coarse.

    If pretrained weights exist, the subclass must also:
    - set `url`, `checksum` and `cache_name`: the URL and the SHA256 checksum of the checkpoint,
      and the name of its folder in the torch-em cache directory.
    - implement `load_pretrained`.

    A scale factor is an integer or a list with one integer for each spatial axis (Z, Y, X).

    Args:
        in_channels: The number of input channels.
    """
    url: Optional[str] = None
    checksum: Optional[str] = None
    cache_name: Optional[str] = None

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.return_outputs = True
        self.features: List[int] = []
        self.scale_factors: List[Union[int, List[int]]] = []
        self.stem_scale_factor: Union[int, List[int]] = 1

    @property
    def out_channels(self):
        """The number of channels of the bottleneck."""
        return self.features[-1]

    def __len__(self):
        return len(self.scale_factors)

    @classmethod
    def get_checkpoint(cls, checkpoint_path: Optional[Union[str, os.PathLike]] = None, download: bool = True) -> str:
        """Get the filepath to the pretrained checkpoint.

        If you pass `checkpoint_path`, this method uses this file.
        Otherwise, it uses the checkpoint in the torch-em cache directory, and it downloads the checkpoint if needed.
        See `torch_em.util.util.get_cache_directory` for the cache directory.

        Args:
            checkpoint_path: The filepath to a checkpoint on disk.
            download: Whether to download the checkpoint if it is not in the cache yet.

        Returns:
            The filepath to the checkpoint.
        """
        if checkpoint_path is not None:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"The checkpoint {checkpoint_path} does not exist.")
            return str(checkpoint_path)

        if cls.url is None:
            raise NotImplementedError(f"The encoder {cls.__name__} does not have pretrained weights.")

        import pooch  # We import pooch here, so that 'import torch_em' does not need it.

        folder = os.path.join(get_cache_directory(), "models", cls.cache_name)
        fname = cls.url.split("/")[-1]
        if not download and not os.path.exists(os.path.join(folder, fname)):
            raise RuntimeError(f"The checkpoint is not in {folder}. Set download=True to download it.")

        return pooch.retrieve(url=cls.url, known_hash=f"sha256:{cls.checksum}", fname=fname, path=folder)

    def load_pretrained(self, checkpoint: Union[str, os.PathLike, OrderedDict]) -> None:
        """Load the pretrained weights into the encoder.

        Args:
            checkpoint: The filepath to a checkpoint or a state dict for the encoder.
        """
        raise NotImplementedError(f"The encoder {type(self).__name__} does not have pretrained weights.")


def load_cryosiam_encoder_state(checkpoint: Union[str, os.PathLike]) -> OrderedDict:
    """Load the encoder weights from a CryoSiam checkpoint.

    Args:
        checkpoint: The filepath to the CryoSiam checkpoint.

    Returns:
        The state dict for the MONAI ResNet of `CryoSiamEncoder`.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    prefix = "_model.encoder."
    # The 'fc' layer belongs to the global projection head of the pretraining.
    return OrderedDict(
        (k[len(prefix):], v) for k, v in state.items() if k.startswith(prefix) and not k.startswith(prefix + "fc.")
    )


@register_encoder("cryosiam")
class CryoSiamEncoder(CryoZooEncoder):
    """The 3D ResNet encoder of CryoSiam, which uses the ResNet from MONAI.

    It has a 7x7x7 stem without stride and four stages with one basic block each.
    The first three stages give the skip connections and the last stage gives the bottleneck.

    Args:
        in_channels: The number of input channels. The pretrained weights expect one channel.
    """
    url = CRYOSIAM_URL
    checksum = CRYOSIAM_CHECKSUM
    cache_name = "cryosiam"

    def __init__(self, in_channels: int = 1):
        super().__init__(in_channels=in_channels)
        if ResNet is None:
            raise RuntimeError("The CryoSiam encoder needs MONAI. Install it from conda-forge or with pip.")

        self.features = [64, 128, 256, 512]
        self.scale_factors = [2, 2, 2]
        self.stem_scale_factor = 1

        # We pin all the settings that the CryoSiam checkpoint depends on, in case the MONAI defaults change.
        self.resnet = ResNet(
            block="basic", layers=[1, 1, 1, 1], block_inplanes=self.features, spatial_dims=3,
            n_input_channels=in_channels, conv1_t_size=7, conv1_t_stride=1, no_max_pool=True, shortcut_type="B",
            feed_forward=False, bias_downsample=True, act=("relu", {"inplace": False}), norm="batch",
        )

    def load_pretrained(self, checkpoint: Union[str, os.PathLike, OrderedDict]) -> None:
        state = checkpoint if isinstance(checkpoint, OrderedDict) else load_cryosiam_encoder_state(checkpoint)
        self.resnet.load_state_dict(state)

    def forward(self, x):
        """Apply the encoder to the input data.

        Args:
            x: The input tensor with the shape (B, C, Z, Y, X).

        Returns:
            The bottleneck, and the list of skip connections from fine to coarse if `return_outputs` is True.
        """
        net = self.resnet
        x = net.act(net.bn1(net.conv1(x)))
        encoder_out = []
        for layer in (net.layer1, net.layer2, net.layer3):
            x = layer(x)
            encoder_out.append(x)
        x = net.layer4(x)

        if self.return_outputs:
            return x, encoder_out
        else:
            return x


class OutputHead(nn.Module):
    """@private
    """
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        self.out_channels = out_channels
        self.upsampler = Upsampler3d(scale_factor, in_channels, in_channels)
        self.conv = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(self.upsampler(x))


class CryoZooUNet(UNetBase):
    """A 3D U-Net with a pretrained CNN encoder and a convolutional decoder from torch-em.

    The decoder always uses the skip connections of the encoder.
    Each spatial input dimension must be divisible by the total downsampling factor of the encoder.
    For the CryoSiam encoder, this factor is 8.
    The pretrained CryoSiam weights expect tomograms with inverted contrast and intensities scaled to [0, 1].

    `init_kwargs` does not contain `encoder_checkpoint`. When torch-em restores a trained model,
    it loads all the weights from the saved training state. Thus it does not load the pretrained weights again.

    Args:
        out_channels: The number of output channels.
        encoder: The name of the encoder. See `ENCODERS` for the supported names.
        in_channels: The number of input channels. The pretrained CryoSiam weights expect one channel.
        encoder_checkpoint: The filepath to a checkpoint or a state dict for the encoder.
            The model loads the encoder weights from it. Use `get_cryozoounet` to get the pretrained weights.
            If you do not pass it, the encoder starts with random weights.
        final_activation: The activation applied after the output convolution.
        postprocessing: A postprocessing function to apply after the U-Net output.
        check_shape: Whether to check the input shape to the U-Net forward call.
        conv_block_kwargs: The keyword arguments for the convolutional blocks of the decoder.
    """
    def __init__(
        self,
        out_channels: int,
        encoder: str = "cryosiam",
        in_channels: int = 1,
        encoder_checkpoint: Optional[Union[str, os.PathLike, OrderedDict]] = None,
        final_activation: Optional[Union[nn.Module, str]] = None,
        postprocessing: Optional[Union[nn.Module, str]] = None,
        check_shape: bool = True,
        **conv_block_kwargs,
    ):
        if encoder not in ENCODERS:
            raise ValueError(f"The encoder '{encoder}' is not supported. Choose one of {list(ENCODERS)}.")
        encoder_model = ENCODERS[encoder](in_channels=in_channels)

        features_decoder = encoder_model.features[::-1]
        decoder = Decoder(
            features=features_decoder,
            scale_factors=encoder_model.scale_factors[::-1],
            conv_block_impl=ConvBlock3d,
            sampler_impl=Upsampler3d,
            skip_channels=features_decoder[1:],
            **conv_block_kwargs,
        )

        if np.all(np.array(encoder_model.stem_scale_factor) == 1):
            out_conv = nn.Conv3d(features_decoder[-1], out_channels, 1)
        else:
            out_conv = OutputHead(features_decoder[-1], out_channels, encoder_model.stem_scale_factor)

        super().__init__(
            encoder=encoder_model,
            base=nn.Identity(),
            decoder=decoder,
            out_conv=out_conv,
            final_activation=final_activation,
            postprocessing=postprocessing,
            check_shape=check_shape,
        )
        self.init_kwargs = {
            "out_channels": out_channels, "encoder": encoder, "in_channels": in_channels,
            "final_activation": final_activation, "postprocessing": postprocessing, "check_shape": check_shape,
            **conv_block_kwargs,
        }

        if encoder_checkpoint is not None:
            self.encoder.load_pretrained(encoder_checkpoint)

    def _check_shape(self, x):
        spatial_shape = tuple(x.shape)[2:]
        factor = np.ones(len(spatial_shape), dtype=int) * np.array(self.encoder.stem_scale_factor)
        for scale_factor in self.encoder.scale_factors:
            factor = factor * np.array(scale_factor)
        if any(sh % fac != 0 for sh, fac in zip(spatial_shape, factor)):
            raise ValueError(f"The input shape {spatial_shape} must be divisible by {factor.tolist()}.")


def get_cryozoounet(
    out_channels: int,
    encoder: str = "cryosiam",
    pretrained: bool = True,
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
    **kwargs,
) -> CryoZooUNet:
    """Get a CryoZooUNet with a pretrained CNN encoder.

    Args:
        out_channels: The number of output channels.
        encoder: The name of the encoder. See `ENCODERS` for the supported names.
        pretrained: Whether to initialize the encoder with its pretrained weights.
        checkpoint_path: The filepath to a checkpoint on disk.
            If you do not pass it, the function downloads the pretrained weights to the torch-em cache directory
            when you use them the first time.
        kwargs: Additional keyword arguments for `CryoZooUNet`.

    Returns:
        The CryoZooUNet.
    """
    if encoder not in ENCODERS:
        raise ValueError(f"The encoder '{encoder}' is not supported. Choose one of {list(ENCODERS)}.")
    encoder_checkpoint = ENCODERS[encoder].get_checkpoint(checkpoint_path) if pretrained else None
    return CryoZooUNet(out_channels=out_channels, encoder=encoder, encoder_checkpoint=encoder_checkpoint, **kwargs)
