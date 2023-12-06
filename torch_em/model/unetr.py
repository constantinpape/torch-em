import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from .unet import Decoder, ConvBlock2d, Upsampler2d
from .vit import get_vision_transformer

try:
    from micro_sam.util import get_sam_model
except ImportError:
    get_sam_model = None


#
# UNETR IMPLEMENTATION [Vision Transformer (ViT from MAE / ViT from SAM) + UNet Decoder from `torch_em`]
#


class UNETR(nn.Module):
    def __init__(
        self,
        backbone="sam",
        encoder="vit_b",
        decoder=None,
        out_channels=1,
        use_sam_stats=False,
        use_mae_stats=False,
        encoder_checkpoint_path=None,
        final_activation=None,
    ) -> None:
        super().__init__()

        self.use_sam_stats = use_sam_stats
        self.use_mae_stats = use_mae_stats

        print(f"Using {encoder} from {backbone.upper()}")

        self.encoder = get_vision_transformer(backbone=backbone, model=encoder)

        if encoder_checkpoint_path is not None:
            if backbone == "sam":
                _, model = get_sam_model(
                    model_type=encoder,
                    checkpoint_path=encoder_checkpoint_path,
                    return_sam=True
                )
                for param1, param2 in zip(model.parameters(), self.encoder.parameters()):
                    param2.data = param1
            elif backbone == "mae":
                raise NotImplementedError

        # parameters for the decoder network
        depth = 3
        initial_features = 64
        gain = 2
        features_decoder = [initial_features * gain ** i for i in range(depth + 1)][::-1]
        scale_factors = depth * [2]
        self.out_channels = out_channels

        if decoder is None:
            self.decoder = Decoder(
                features=features_decoder,
                scale_factors=scale_factors[::-1],
                conv_block_impl=ConvBlock2d,
                sampler_impl=Upsampler2d
            )
        else:
            self.decoder = decoder

        self.z_inputs = ConvBlock2d(self.encoder.in_chans, features_decoder[-1])

        self.base = ConvBlock2d(self.encoder.embed_dim, features_decoder[0])
        self.out_conv = nn.Conv2d(features_decoder[-1], out_channels, 1)

        self.deconv1 = Deconv2DBlock(self.encoder.embed_dim, features_decoder[0])
        self.deconv2 = Deconv2DBlock(features_decoder[0], features_decoder[1])
        self.deconv3 = Deconv2DBlock(features_decoder[1], features_decoder[2])

        self.deconv4 = SingleDeconv2DBlock(features_decoder[-1], features_decoder[-1])

        self.decoder_head = ConvBlock2d(2*features_decoder[-1], features_decoder[-1])
        self.final_activation = self._get_activation(final_activation)

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

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.use_sam_stats:
            pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(device)
            pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(device)
        elif self.use_mae_stats:
            # TODO: add mean std from mae experiments (or open up arguments for this)
            raise NotImplementedError
        else:
            pixel_mean = torch.Tensor([0.0, 0.0, 0.0]).view(-1, 1, 1).to(device)
            pixel_std = torch.Tensor([1.0, 1.0, 1.0]).view(-1, 1, 1).to(device)

        x = (x - pixel_mean) / pixel_std
        h, w = x.shape[-2:]
        padh = self.encoder.img_size - h
        padw = self.encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (self.encoder.img_size, self.encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def forward(self, x):
        org_shape = x.shape[-2:]

        # backbone used for reshaping inputs to the desired "encoder" shape
        x = torch.stack([self.preprocess(e) for e in x], dim=0)

        z0 = self.z_inputs(x)

        z12, from_encoder = self.encoder(x)
        x = self.base(z12)

        from_encoder = from_encoder[::-1]
        z9 = self.deconv1(from_encoder[0])

        z6 = self.deconv1(from_encoder[1])
        z6 = self.deconv2(z6)

        z3 = self.deconv1(from_encoder[2])
        z3 = self.deconv2(z3)
        z3 = self.deconv3(z3)

        updated_from_encoder = [z9, z6, z3]
        x = self.decoder(x, encoder_inputs=updated_from_encoder)
        x = self.deconv4(x)
        x = torch.cat([x, z0], dim=1)

        x = self.decoder_head(x)

        x = self.out_conv(x)
        if self.final_activation is not None:
            x = self.final_activation(x)

        x = self.postprocess_masks(x, org_shape, org_shape)
        return x


#
#  ADDITIONAL FUNCTIONALITIES
#


class SingleDeconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)


class SingleConv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size - 1) // 2))

    def forward(self, x):
        return self.block(x)


class Conv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv2DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Deconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv2DBlock(in_planes, out_planes),
            SingleConv2DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)
