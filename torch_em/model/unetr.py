import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from functools import partial
from torch_em.model.unet import Decoder, ConvBlock2d, Upsampler2d

# we catch ImportErrors here because segment_anything and micro_sam should
# only be optional dependencies for torch_em
try:
    from segment_anything.modeling import ImageEncoderViT
    _sam_import_success = True
except ImportError:
    ImageEncoderViT = object
    _sam_import_success = False

try:
    from micro_sam.util import get_sam_model
except ImportError:
    get_sam_model = None


class ViTb_Sam(ImageEncoderViT):
    """Vision Transformer derived from the Segment Anything Codebase (https://arxiv.org/abs/2304.02643):
    https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py
    """
    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 768,
        global_attn_indexes: Tuple[int, ...] = ...,
        **kwargs
    ) -> None:
        if not _sam_import_success:
            raise RuntimeError(
                "The vision transformer backend can only be initialized if segment anything is installed."
                "Please install segment anything from https://github.com/facebookresearch/segment-anything."
                "and then rerun your code."
            )

        super().__init__(**kwargs)
        self.chunks_for_projection = global_attn_indexes
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        list_from_encoder = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.chunks_for_projection:
                list_from_encoder.append(x)

        x = x.permute(0, 3, 1, 2)
        list_from_encoder = [e.permute(0, 3, 1, 2) for e in list_from_encoder]
        return x, list_from_encoder[:3]  # type: ignore


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


class UNETR(nn.Module):
    def __init__(
        self,
        encoder=None,
        decoder=None,
        out_channels=1
    ) -> None:
        depth = 3
        initial_features = 64
        gain = 2
        features_decoder = [initial_features * gain ** i for i in range(depth + 1)][::-1]
        scale_factors = depth * [2]
        self.out_channels = out_channels

        super().__init__()

        if encoder is None:
            self.encoder = ViTb_Sam(
                depth=12,
                embed_dim=768,
                img_size=1024,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),  # type: ignore
                num_heads=12,
                patch_size=16,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=[2, 5, 8, 11],  # type: ignore
                window_size=14,
                out_chans=256,
            )
        else:
            self.encoder = encoder

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
        self.final_activation = nn.Sigmoid()

        self.deconv1 = Deconv2DBlock(self.encoder.embed_dim, features_decoder[0])
        self.deconv2 = Deconv2DBlock(features_decoder[0], features_decoder[1])
        self.deconv3 = Deconv2DBlock(features_decoder[1], features_decoder[2])

        self.deconv4 = SingleDeconv2DBlock(features_decoder[-1], features_decoder[-1])

        self.decoder_head = ConvBlock2d(2*features_decoder[-1], features_decoder[-1])

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # TODO Why do we have hard-code image normalization here???
        pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(device)
        pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(device)

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

        x = torch.stack([self.preprocess(e) for e in x], dim=0)

        z0 = self.z_inputs(x)

        z12, from_encoder = self.encoder(x)
        x = self.base(z12)

        from_encoder = from_encoder[::-1]
        z9 = self.deconv1(from_encoder[0])

        z6 = self.deconv1(from_encoder[1])
        z6 = self.deconv2(z6)

        z3 = self.deconv1(from_encoder[1])
        z3 = self.deconv2(z3)
        z3 = self.deconv3(z3)

        updated_from_encoder = [z9, z6, z3]
        x = self.decoder(x, encoder_inputs=updated_from_encoder)
        x = self.deconv4(x)
        x = torch.cat([x, z0], dim=1)

        x = self.decoder_head(x)

        x = self.out_conv(x)
        x = self.final_activation(x)

        x = self.postprocess_masks(x, org_shape, org_shape)
        return x


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


def build_unetr_with_sam_intialization(out_channels=1, model_type="vit_b", checkpoint_path=None):
    if get_sam_model is None:
        raise RuntimeError(
            "micro_sam is required to initialize the UNETR image encoder from segment anything weights."
            "Please install it from"
            "and then rerun your code."
        )
    predictor = get_sam_model(model_type=model_type, checkpoint_path=checkpoint_path)
    _image_encoder = predictor.model.image_encoder

    image_encoder = ViTb_Sam()
    # FIXME this doesn't work yet because the parameters don't match
    with torch.no_grad():
        for param1, param2 in zip(_image_encoder.parameters(), image_encoder.parameters()):
            param2.data = param1.data

    unetr = UNETR(encoder=image_encoder, out_channels=out_channels)
    return unetr
