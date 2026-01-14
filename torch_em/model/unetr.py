from functools import partial
from collections import OrderedDict
from typing import Optional, Tuple, Union, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import get_vision_transformer
from .unet import Decoder, ConvBlock2d, ConvBlock3d, Upsampler2d, Upsampler3d, _update_conv_kwargs

try:
    from micro_sam.util import get_sam_model
except ImportError:
    get_sam_model = None

try:
    from micro_sam2.util import get_sam2_model
except ImportError:
    get_sam2_model = None

try:
    from micro_sam3.util import get_sam3_model
except ImportError:
    get_sam3_model = None


#
# UNETR IMPLEMENTATION [Vision Transformer (ViT from SAM / SAM2 / SAM3 / DINOv2 / DINOv3 / MAE / ScaleMAE) + UNet Decoder from `torch_em`]  # noqa
#


class UNETRBase(nn.Module):
    """Base class for implementing a UNETR.

    Args:
        img_size: The size of the input for the image encoder. Input images will be resized to match this size.
        backbone: The name of the vision transformer implementation.
            One of "sam", "sam2", "sam3, "mae", "scalemae", "dinov2", "dinov3".
        encoder: The vision transformer. Can either be a name, such as "vit_b" or a torch module.
        decoder: The convolutional decoder.
        out_channels: The number of output channels of the UNETR.
        use_sam_stats: Whether to normalize the input data with the statistics of the
            pretrained SAM / SAM2 / SAM3 model.
        use_dino_stats: Whether to normalize the input data with the statistics of the
            pretrained DINOv2 / DINOv3 model.
        use_mae_stats: Whether to normalize the input data with the statistics of the pretrained MAE model.
        resize_input: Whether to resize the input images to match `img_size`.
            By default, it resizes the inputs to match the `img_size`.
        encoder_checkpoint: Checkpoint for initializing the vision transformer.
            Can either be a filepath or an already loaded checkpoint.
        final_activation: The activation to apply to the UNETR output.
        use_skip_connection: Whether to use skip connections. By default, it uses skip connections.
        embed_dim: The embedding dimensionality, corresponding to the output dimension of the vision transformer.
        use_conv_transpose: Whether to use transposed convolutions instead of resampling for upsampling.
            By default, it uses resampling for upsampling.

        NOTE: The currently supported combinations of 'backbone' x 'encoder' (in same order) combinations as following

        SAM_family_models:
            - 'sam' x 'vit_b'
            - 'sam' x 'vit_l'
            - 'sam' x 'vit_h'
            - 'sam2' x 'hvit_t'
            - 'sam2' x 'hvit_s'
            - 'sam2' x 'hvit_b'
            - 'sam2' x 'hvit_l'
            - 'sam3' x 'vit_pe'

        DINO_family_models:
            - 'dinov2' x 'vit_s'
            - 'dinov2' x 'vit_b'
            - 'dinov2' x 'vit_l'
            - 'dinov2' x 'vit_g'
            - 'dinov2' x 'vit_s_reg4'
            - 'dinov2' x 'vit_b_reg4'
            - 'dinov2' x 'vit_l_reg4'
            - 'dinov2' x 'vit_g_reg4'
            - 'dinov3' x 'vit_s'
            - 'dinov3' x 'vit_s+'
            - 'dinov3' x 'vit_b'
            - 'dinov3' x 'vit_l'
            - 'dinov3' x 'vit_l+'
            - 'dinov3' x 'vit_h+'
            - 'dinov3' x 'vit_7b'

        MAE_family_models:
            - 'mae' x 'vit_b'
            - 'mae' x 'vit_l'
            - 'mae' x 'vit_h'
            - 'scalemae' x 'vit_b'
            - 'scalemae' x 'vit_l'
            - 'scalemae' x 'vit_h'
    """
    def __init__(
        self,
        img_size: int = 1024,
        backbone: Literal["sam", "sam2", "sam3", "mae", "scalemae", "dinov2", "dinov3"] = "sam",
        encoder: Optional[Union[nn.Module, str]] = "vit_b",
        decoder: Optional[nn.Module] = None,
        out_channels: int = 1,
        use_sam_stats: bool = False,
        use_mae_stats: bool = False,
        use_dino_stats: bool = False,
        resize_input: bool = True,
        encoder_checkpoint: Optional[Union[str, OrderedDict]] = None,
        final_activation: Optional[Union[str, nn.Module]] = None,
        use_skip_connection: bool = True,
        embed_dim: Optional[int] = None,
        use_conv_transpose: bool = False,
        **kwargs
    ) -> None:
        super().__init__()

        self.img_size = img_size
        self.use_sam_stats = use_sam_stats
        self.use_mae_stats = use_mae_stats
        self.use_dino_stats = use_dino_stats
        self.use_skip_connection = use_skip_connection
        self.resize_input = resize_input
        self.use_conv_transpose = use_conv_transpose
        self.backbone = backbone

        if isinstance(encoder, str):  # e.g. "vit_b" / "hvit_b" / "vit_pe"
            print(f"Using {encoder} from {backbone.upper()}")
            self.encoder = get_vision_transformer(img_size=img_size, backbone=backbone, model=encoder, **kwargs)

            if encoder_checkpoint is not None:
                self._load_encoder_from_checkpoint(backbone=backbone, encoder=encoder, checkpoint=encoder_checkpoint)

            if embed_dim is None:
                embed_dim = self.encoder.embed_dim

        else:  # `nn.Module` ViT backbone
            self.encoder = encoder

            have_neck = False
            for name, _ in self.encoder.named_parameters():
                if name.startswith("neck"):
                    have_neck = True

            if embed_dim is None:
                if have_neck:
                    embed_dim = self.encoder.neck[2].out_channels  # the value is 256
                else:
                    embed_dim = self.encoder.patch_embed.proj.out_channels

        self.embed_dim = embed_dim
        self.final_activation = self._get_activation(final_activation)

    def _load_encoder_from_checkpoint(self, backbone, encoder, checkpoint):
        """Function to load pretrained weights to the image encoder.
        """
        if isinstance(checkpoint, str):
            if backbone == "sam" and isinstance(encoder, str):
                # If we have a SAM encoder, then we first try to load the full SAM Model
                # (using micro_sam) and otherwise fall back on directly loading the encoder state
                # from the checkpoint
                try:
                    _, model = get_sam_model(model_type=encoder, checkpoint_path=checkpoint, return_sam=True)
                    encoder_state = model.image_encoder.state_dict()
                except Exception:
                    # Try loading the encoder state directly from a checkpoint.
                    encoder_state = torch.load(checkpoint, weights_only=False)

            elif backbone == "sam2" and isinstance(encoder, str):
                # If we have a SAM2 encoder, then we first try to load the full SAM2 Model.
                # (using micro_sam2) and otherwise fall back on directly loading the encoder state
                # from the checkpoint
                try:
                    model = get_sam2_model(model_type=encoder, checkpoint_path=checkpoint)
                    encoder_state = model.image_encoder.state_dict()
                except Exception:
                    # Try loading the encoder state directly from a checkpoint.
                    encoder_state = torch.load(checkpoint, weights_only=False)

            elif backbone == "sam3" and isinstance(encoder, str):
                # If we have a SAM3 encoder, then we first try to load the full SAM3 Model.
                # (using micro_sam3) and otherwise fall back on directly loading the encoder state
                # from the checkpoint
                try:
                    model = get_sam3_model(checkpoint_path=checkpoint)
                    encoder_state = model.backbone.vision_backbone.state_dict()
                    # Let's align loading the encoder weights with expected parameter names
                    encoder_state = {
                        k[len("trunk."):] if k.startswith("trunk.") else k: v for k, v in encoder_state.items()
                    }
                    # And drop the 'convs' and 'sam2_convs' - these seem like some upsampling blocks.
                    encoder_state = {
                        k: v for k, v in encoder_state.items()
                        if not (k.startswith("convs.") or k.startswith("sam2_convs."))
                    }
                except Exception:
                    # Try loading the encoder state directly from a checkpoint.
                    encoder_state = torch.load(checkpoint, weights_only=False)

            elif backbone == "mae":
                # vit initialization hints from:
                #     - https://github.com/facebookresearch/mae/blob/main/main_finetune.py#L233-L242
                encoder_state = torch.load(checkpoint, weights_only=False)["model"]
                encoder_state = OrderedDict({
                    k: v for k, v in encoder_state.items() if (k != "mask_token" and not k.startswith("decoder"))
                })
                # Let's remove the `head` from our current encoder (as the MAE pretrained don't expect it)
                current_encoder_state = self.encoder.state_dict()
                if ("head.weight" in current_encoder_state) and ("head.bias" in current_encoder_state):
                    del self.encoder.head

            elif backbone == "scalemae":
                # Load the encoder state directly from a checkpoint.
                encoder_state = torch.load(checkpoint)["model"]
                encoder_state = OrderedDict({
                    k: v for k, v in encoder_state.items()
                    if not k.startswith(("mask_token", "decoder", "fcn", "fpn", "pos_embed"))
                })

                # Let's remove the `head` from our current encoder (as the MAE pretrained don't expect it)
                current_encoder_state = self.encoder.state_dict()
                if ("head.weight" in current_encoder_state) and ("head.bias" in current_encoder_state):
                    del self.encoder.head

                if "pos_embed" in current_encoder_state:  # NOTE: ScaleMAE uses 'pos. embeddings' in a diff. format.
                    del self.encoder.pos_embed

            elif backbone in ["dinov2", "dinov3"]:  # Load the encoder state directly from a checkpoint.
                encoder_state = torch.load(checkpoint)

            else:
                raise ValueError(
                    f"We don't support either the '{backbone}' backbone or the '{encoder}' model combination (or both)."
                )

        else:
            encoder_state = checkpoint

        self.encoder.load_state_dict(encoder_state)

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

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """Compute the output size given input size and target long side length.

        Args:
            oldh: The input image height.
            oldw: The input image width.
            long_side_length: The longest side length for resizing.

        Returns:
            The new image height.
            The new image width.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def resize_longest_side(self, image: torch.Tensor) -> torch.Tensor:
        """Resize the image so that the longest side has the correct length.

        Expects batched images with shape BxCxHxW OR BxCxDxHxW and float format.

        Args:
            image: The input image.

        Returns:
            The resized image.
        """
        if image.ndim == 4:  # i.e. 2d image
            target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.encoder.img_size)
            return F.interpolate(image, target_size, mode="bilinear", align_corners=False, antialias=True)
        elif image.ndim == 5:  # i.e. 3d volume
            B, C, Z, H, W = image.shape
            target_size = self.get_preprocess_shape(H, W, self.img_size)
            return F.interpolate(image, (Z, *target_size), mode="trilinear", align_corners=False)
        else:
            raise ValueError("Expected 4d or 5d inputs, got", image.shape)

    def _as_stats(self, mean, std, device, dtype, is_3d: bool):
        """@private
        """
        # Either 2d batch: (1, C, 1, 1) or 3d batch: (1, C, 1, 1, 1).
        view_shape = (1, -1, 1, 1, 1) if is_3d else (1, -1, 1, 1)
        pixel_mean = torch.tensor(mean, device=device, dtype=dtype).view(*view_shape)
        pixel_std = torch.tensor(std, device=device, dtype=dtype).view(*view_shape)
        return pixel_mean, pixel_std

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """@private
        """
        device = x.device
        is_3d = (x.ndim == 5)
        device, dtype = x.device, x.dtype

        if self.use_sam_stats:
            mean, std = (123.675, 116.28, 103.53), (58.395, 57.12, 57.375)
        elif self.use_mae_stats:  # TODO: add mean std from mae / scalemae experiments (or open up arguments for this)
            raise NotImplementedError
        elif self.use_dino_stats or (self.use_sam_stats and self.backbone == "sam2"):
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        elif self.use_sam_stats and self.backbone == "sam3":
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        else:
            mean, std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)

        pixel_mean, pixel_std = self._as_stats(mean, std, device=device, dtype=dtype, is_3d=is_3d)

        if self.resize_input:
            x = self.resize_longest_side(x)
        input_shape = x.shape[-3:] if is_3d else x.shape[-2:]

        x = (x - pixel_mean) / pixel_std
        h, w = x.shape[-2:]
        padh = self.encoder.img_size - h
        padw = self.encoder.img_size - w

        if is_3d:
            x = F.pad(x, (0, padw, 0, padh, 0, 0))
        else:
            x = F.pad(x, (0, padw, 0, padh))

        return x, input_shape

    def postprocess_masks(
        self, masks: torch.Tensor, input_size: Tuple[int, ...], original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """@private
        """
        if masks.ndim == 4:  # i.e. 2d labels
            masks = F.interpolate(
                masks,
                (self.encoder.img_size, self.encoder.img_size),
                mode="bilinear",
                align_corners=False,
            )
            masks = masks[..., : input_size[0], : input_size[1]]
            masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)

        elif masks.ndim == 5:  # i.e. 3d volumetric labels
            masks = F.interpolate(
                masks,
                (input_size[0], self.img_size, self.img_size),
                mode="trilinear",
                align_corners=False,
            )
            masks = masks[..., :input_size[0], :input_size[1], :input_size[2]]
            masks = F.interpolate(masks, original_size, mode="trilinear", align_corners=False)

        else:
            raise ValueError("Expected 4d or 5d labels, got", masks.shape)

        return masks


class UNETR(UNETRBase):
    """A (2d-only) UNet Transformer using a vision transformer as encoder and a convolutional decoder.
    """
    def __init__(
        self,
        img_size: int = 1024,
        backbone: Literal["sam", "sam2", "sam3", "mae", "scalemae", "dinov2", "dinov3"] = "sam",
        encoder: Optional[Union[nn.Module, str]] = "vit_b",
        decoder: Optional[nn.Module] = None,
        out_channels: int = 1,
        use_sam_stats: bool = False,
        use_mae_stats: bool = False,
        use_dino_stats: bool = False,
        resize_input: bool = True,
        encoder_checkpoint: Optional[Union[str, OrderedDict]] = None,
        final_activation: Optional[Union[str, nn.Module]] = None,
        use_skip_connection: bool = True,
        embed_dim: Optional[int] = None,
        use_conv_transpose: bool = False,
        **kwargs
    ) -> None:

        super().__init__(
            img_size=img_size,
            backbone=backbone,
            encoder=encoder,
            decoder=decoder,
            out_channels=out_channels,
            use_sam_stats=use_sam_stats,
            use_mae_stats=use_mae_stats,
            use_dino_stats=use_dino_stats,
            resize_input=resize_input,
            encoder_checkpoint=encoder_checkpoint,
            final_activation=final_activation,
            use_skip_connection=use_skip_connection,
            embed_dim=embed_dim,
            use_conv_transpose=use_conv_transpose,
            **kwargs,
        )

        encoder = self.encoder

        if backbone == "sam2" and hasattr(encoder, "trunk"):
            in_chans = encoder.trunk.patch_embed.proj.in_channels
        elif hasattr(encoder, "in_chans"):
            in_chans = encoder.in_chans
        else:  # `nn.Module` ViT backbone.
            try:
                in_chans = encoder.patch_embed.proj.in_channels
            except AttributeError:  # for getting the input channels while using 'vit_t' from MobileSam
                in_chans = encoder.patch_embed.seq[0].c.in_channels

        # parameters for the decoder network
        depth = 3
        initial_features = 64
        gain = 2
        features_decoder = [initial_features * gain ** i for i in range(depth + 1)][::-1]
        scale_factors = depth * [2]
        self.out_channels = out_channels

        # choice of upsampler - to use (bilinear interpolation + conv) or conv transpose
        _upsampler = SingleDeconv2DBlock if use_conv_transpose else Upsampler2d

        self.decoder = decoder or Decoder(
            features=features_decoder,
            scale_factors=scale_factors[::-1],
            conv_block_impl=ConvBlock2d,
            sampler_impl=_upsampler,
        )

        if use_skip_connection:
            self.deconv1 = Deconv2DBlock(
                in_channels=self.embed_dim,
                out_channels=features_decoder[0],
                use_conv_transpose=use_conv_transpose,
            )
            self.deconv2 = nn.Sequential(
                Deconv2DBlock(
                    in_channels=self.embed_dim,
                    out_channels=features_decoder[0],
                    use_conv_transpose=use_conv_transpose,
                ),
                Deconv2DBlock(
                    in_channels=features_decoder[0],
                    out_channels=features_decoder[1],
                    use_conv_transpose=use_conv_transpose,
                )
            )
            self.deconv3 = nn.Sequential(
                Deconv2DBlock(
                    in_channels=self.embed_dim,
                    out_channels=features_decoder[0],
                    use_conv_transpose=use_conv_transpose,
                ),
                Deconv2DBlock(
                    in_channels=features_decoder[0],
                    out_channels=features_decoder[1],
                    use_conv_transpose=use_conv_transpose,
                ),
                Deconv2DBlock(
                    in_channels=features_decoder[1],
                    out_channels=features_decoder[2],
                    use_conv_transpose=use_conv_transpose,
                )
            )
            self.deconv4 = ConvBlock2d(in_chans, features_decoder[-1])
        else:
            self.deconv1 = Deconv2DBlock(
                in_channels=self.embed_dim,
                out_channels=features_decoder[0],
                use_conv_transpose=use_conv_transpose,
            )
            self.deconv2 = Deconv2DBlock(
                in_channels=features_decoder[0],
                out_channels=features_decoder[1],
                use_conv_transpose=use_conv_transpose,
            )
            self.deconv3 = Deconv2DBlock(
                in_channels=features_decoder[1],
                out_channels=features_decoder[2],
                use_conv_transpose=use_conv_transpose,
            )
            self.deconv4 = Deconv2DBlock(
                in_channels=features_decoder[2],
                out_channels=features_decoder[3],
                use_conv_transpose=use_conv_transpose,
            )

        self.base = ConvBlock2d(self.embed_dim, features_decoder[0])
        self.out_conv = nn.Conv2d(features_decoder[-1], out_channels, 1)
        self.deconv_out = _upsampler(
            scale_factor=2, in_channels=features_decoder[-1], out_channels=features_decoder[-1]
        )
        self.decoder_head = ConvBlock2d(2 * features_decoder[-1], features_decoder[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the UNETR to the input data.

        Args:
            x: The input tensor.

        Returns:
            The UNETR output.
        """
        original_shape = x.shape[-2:]

        # Reshape the inputs to the shape expected by the encoder
        # and normalize the inputs if normalization is part of the model.
        x, input_shape = self.preprocess(x)

        encoder_outputs = self.encoder(x)

        if isinstance(encoder_outputs[-1], list):
            # `encoder_outputs` can be arranged in only two forms:
            #   - either we only return the image embeddings
            #   - or, we return the image embeddings and the "list" of global attention layers
            z12, from_encoder = encoder_outputs
        else:
            z12 = encoder_outputs

        if self.use_skip_connection:
            from_encoder = from_encoder[::-1]
            z9 = self.deconv1(from_encoder[0])
            z6 = self.deconv2(from_encoder[1])
            z3 = self.deconv3(from_encoder[2])
            z0 = self.deconv4(x)

        else:
            z9 = self.deconv1(z12)
            z6 = self.deconv2(z9)
            z3 = self.deconv3(z6)
            z0 = self.deconv4(z3)

        updated_from_encoder = [z9, z6, z3]

        x = self.base(z12)
        x = self.decoder(x, encoder_inputs=updated_from_encoder)
        x = self.deconv_out(x)

        x = torch.cat([x, z0], dim=1)
        x = self.decoder_head(x)

        x = self.out_conv(x)
        if self.final_activation is not None:
            x = self.final_activation(x)

        x = self.postprocess_masks(x, input_shape, original_shape)
        return x


class UNETR2D(UNETR):
    """A two-dimensional UNet Transformer using a vision transformer as encoder and a convolutional decoder.
    """
    pass


class UNETR3D(UNETRBase):
    """A three dimensional UNet Transformer using a vision transformer as encoder and a convolutional decoder.
    """
    def __init__(
        self,
        img_size: int = 1024,
        backbone: Literal["sam", "sam2", "sam3", "mae", "scalemae", "dinov2", "dinov3"] = "sam",
        encoder: Optional[Union[nn.Module, str]] = "hvit_b",
        decoder: Optional[nn.Module] = None,
        out_channels: int = 1,
        use_sam_stats: bool = False,
        use_mae_stats: bool = False,
        use_dino_stats: bool = False,
        resize_input: bool = True,
        encoder_checkpoint: Optional[Union[str, OrderedDict]] = None,
        final_activation: Optional[Union[str, nn.Module]] = None,
        use_skip_connection: bool = False,
        embed_dim: Optional[int] = None,
        use_conv_transpose: bool = False,
        use_strip_pooling: bool = True,
        **kwargs
    ):
        if use_skip_connection:
            raise NotImplementedError("The framework cannot handle skip connections atm.")
        if use_conv_transpose:
            raise NotImplementedError("It's not enabled to switch between interpolation and transposed convolutions.")

        # Sort the `embed_dim` out
        embed_dim = 256 if embed_dim is None else embed_dim

        super().__init__(
            img_size=img_size,
            backbone=backbone,
            encoder=encoder,
            decoder=decoder,
            out_channels=out_channels,
            use_sam_stats=use_sam_stats,
            use_mae_stats=use_mae_stats,
            use_dino_stats=use_dino_stats,
            resize_input=resize_input,
            encoder_checkpoint=encoder_checkpoint,
            final_activation=final_activation,
            use_skip_connection=use_skip_connection,
            embed_dim=embed_dim,
            use_conv_transpose=use_conv_transpose,
            **kwargs,
        )

        # The 3d convolutional decoder.
        # First, get the important parameters for the decoder.
        depth = 3
        initial_features = 64
        gain = 2
        features_decoder = [initial_features * gain ** i for i in range(depth + 1)][::-1]
        scale_factors = [1, 2, 2]
        self.out_channels = out_channels

        # The mapping blocks.
        self.deconv1 = Deconv3DBlock(
            in_channels=embed_dim,
            out_channels=features_decoder[0],
            scale_factor=scale_factors,
            use_strip_pooling=use_strip_pooling,
        )
        self.deconv2 = Deconv3DBlock(
            in_channels=features_decoder[0],
            out_channels=features_decoder[1],
            scale_factor=scale_factors,
            use_strip_pooling=use_strip_pooling,
        )
        self.deconv3 = Deconv3DBlock(
            in_channels=features_decoder[1],
            out_channels=features_decoder[2],
            scale_factor=scale_factors,
            use_strip_pooling=use_strip_pooling,
        )
        self.deconv4 = Deconv3DBlock(
            in_channels=features_decoder[2],
            out_channels=features_decoder[3],
            scale_factor=scale_factors,
            use_strip_pooling=use_strip_pooling,
        )

        # The core decoder block.
        self.decoder = decoder or Decoder(
            features=features_decoder,
            scale_factors=[scale_factors] * depth,
            conv_block_impl=partial(ConvBlock3dWithStrip, use_strip_pooling=use_strip_pooling),
            sampler_impl=Upsampler3d,
        )

        # And the final upsampler to match the expected dimensions.
        self.deconv_out = Deconv3DBlock(  # NOTE: changed `end_up` to `deconv_out`
            in_channels=features_decoder[-1],
            out_channels=features_decoder[-1],
            scale_factor=scale_factors,
            use_strip_pooling=use_strip_pooling,
        )

        # Additional conjunction blocks.
        self.base = ConvBlock3dWithStrip(
            in_channels=embed_dim,
            out_channels=features_decoder[0],
            use_strip_pooling=use_strip_pooling,
        )

        # And the output layers.
        self.decoder_head = ConvBlock3dWithStrip(
            in_channels=2 * features_decoder[-1],
            out_channels=features_decoder[-1],
            use_strip_pooling=use_strip_pooling,
        )
        self.out_conv = nn.Conv3d(features_decoder[-1], out_channels, 1)

    def forward(self, x: torch.Tensor):
        """Forward pass of the UNETR-3D model.

        Args:
            x: Inputs of expected shape (B, C, Z, Y, X), where Z considers flexible inputs.

        Returns:
            The UNETR output.
        """
        B, C, Z, H, W = x.shape
        original_shape = (Z, H, W)

        # Preprocessing step
        x, input_shape = self.preprocess(x)

        # Run the image encoder.
        curr_features = torch.stack([self.encoder(x[:, :, i])[0] for i in range(Z)], dim=2)

        # Prepare the counterparts for the decoder.
        # NOTE: The section below is sequential, there's no skip connections atm.
        z9 = self.deconv1(curr_features)
        z6 = self.deconv2(z9)
        z3 = self.deconv3(z6)
        z0 = self.deconv4(z3)

        updated_from_encoder = [z9, z6, z3]

        # Align the features through the base block.
        x = self.base(curr_features)
        # Run the decoder
        x = self.decoder(x, encoder_inputs=updated_from_encoder)
        x = self.deconv_out(x)  # NOTE before `end_up`

        # And the final output head.
        x = torch.cat([x, z0], dim=1)
        x = self.decoder_head(x)
        x = self.out_conv(x)
        if self.final_activation is not None:
            x = self.final_activation(x)

        # Postprocess the output back to original size.
        x = self.postprocess_masks(x, input_shape, original_shape)
        return x

#
#  ADDITIONAL FUNCTIONALITIES
#


def _strip_pooling_layers(enabled, channels) -> nn.Module:
    return DepthStripPooling(channels) if enabled else nn.Identity()


class DepthStripPooling(nn.Module):
    """@private
    """
    def __init__(self, channels: int, reduction: int = 4):
        """Block for strip pooling along the depth dimension (only).

        eg. for 3D (Z > 1) - it aggregates global context across depth by adaptive avg pooling
        to Z=1, and then passes through a small 1x1x1 MLP, then broadcasts it back to Z to
        modulate the original features (using a gated residual).

        For 2D (Z == 1 or Z == 3): returns input unchanged (no-op).

        Args:
            channels: The output channels.
            reduction: The reduction of the hidden layers.
        """
        super().__init__()
        hidden = max(1, channels // reduction)
        self.conv1 = nn.Conv3d(channels, hidden, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(hidden)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"DepthStripPooling expects 5D tensors as input, got '{x.shape}'.")

        B, C, Z, H, W = x.shape
        if Z == 1 or Z == 3:  # i.e. 2d-as-1-slice or RGB_2d-as-1-slice.
            return x  # We simply do nothing there.

        # We pool only along the depth dimension: i.e. target shape (B, C, 1, H, W)
        feat = F.adaptive_avg_pool3d(x, output_size=(1, H, W))
        feat = self.conv1(feat)
        feat = self.bn1(feat)
        feat = self.relu(feat)
        feat = self.conv2(feat)
        gate = torch.sigmoid(feat).expand(B, C, Z, H, W)  # Broadcast the collapsed depth context back to all slices

        # Gated residual fusion
        return x * gate + x


class Deconv3DBlock(nn.Module):
    """@private
    """
    def __init__(
        self,
        scale_factor,
        in_channels,
        out_channels,
        kernel_size=3,
        anisotropic_kernel=True,
        use_strip_pooling=True,
    ):
        super().__init__()
        conv_block_kwargs = {
            "in_channels": out_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "padding": ((kernel_size - 1) // 2),
        }
        if anisotropic_kernel:
            conv_block_kwargs = _update_conv_kwargs(conv_block_kwargs, scale_factor)

        self.block = nn.Sequential(
            Upsampler3d(scale_factor, in_channels, out_channels),
            nn.Conv3d(**conv_block_kwargs),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
            _strip_pooling_layers(enabled=use_strip_pooling, channels=out_channels),
        )

    def forward(self, x):
        return self.block(x)


class ConvBlock3dWithStrip(nn.Module):
    """@private
    """
    def __init__(
        self, in_channels: int, out_channels: int, use_strip_pooling: bool = True, **kwargs
    ):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock3d(in_channels, out_channels, **kwargs),
            _strip_pooling_layers(enabled=use_strip_pooling, channels=out_channels),
        )

    def forward(self, x):
        return self.block(x)


class SingleDeconv2DBlock(nn.Module):
    """@private
    """
    def __init__(self, scale_factor, in_channels, out_channels):
        super().__init__()
        self.block = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)


class SingleConv2DBlock(nn.Module):
    """@private
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.block = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=((kernel_size - 1) // 2)
        )

    def forward(self, x):
        return self.block(x)


class Conv2DBlock(nn.Module):
    """@private
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv2DBlock(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Deconv2DBlock(nn.Module):
    """@private
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, use_conv_transpose=True):
        super().__init__()
        _upsampler = SingleDeconv2DBlock if use_conv_transpose else Upsampler2d
        self.block = nn.Sequential(
            _upsampler(scale_factor=2, in_channels=in_channels, out_channels=out_channels),
            SingleConv2DBlock(out_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)
