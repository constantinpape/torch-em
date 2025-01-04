from typing import Tuple
from functools import partial

import torch
import torch.nn as nn

# we catch ImportErrors here because segment_anything, micro_sam and timm should
# only be optional dependencies for torch_em
try:
    from segment_anything.modeling import ImageEncoderViT
    _sam_import_success = True
except ImportError:
    ImageEncoderViT = object
    _sam_import_success = False

try:
    from timm.models.vision_transformer import VisionTransformer
    _timm_import_success = True
except ImportError:
    VisionTransformer = object
    _timm_import_success = False


class ViT_Sam(ImageEncoderViT):
    """Vision Transformer derived from the Segment Anything Codebase (https://arxiv.org/abs/2304.02643).

    Based on:
    https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py

    Args:
        in_chans: The number of input channels.
        embed_dim: The embedding dimension, corresponding to the number of output channels of the vision transformer.
        global_attn_indexes: The global attention indices.
        kwargs: Keyword arguments for the image encoder base class.
    """
    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 768,
        global_attn_indexes: Tuple[int, ...] = ...,
        **kwargs,
    ) -> None:
        if not _sam_import_success:
            raise RuntimeError(
                "The vision transformer backend can only be initialized if segment anything is installed."
                "Please install segment anything from https://github.com/facebookresearch/segment-anything."
                "and then rerun your code."
            )

        super().__init__(
            embed_dim=embed_dim,
            global_attn_indexes=global_attn_indexes,
            **kwargs,
        )
        self.chunks_for_projection = global_attn_indexes
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the vision transformer to input data.

        Args:
            x: The input data.

        Returns:
            The vision transformer output.
        """
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
        return x, list_from_encoder[:3]


class ViT_MAE(VisionTransformer):
    """Vision Transformer derived from the Masked Auto Encoder Codebase (https://arxiv.org/abs/2111.06377).

    Based on:
    https://github.com/facebookresearch/mae/blob/main/models_vit.py#L20-L53

    Args:
        img_size: The size of the input for the image encoder. Input images will be resized to match this size.
        in_chans: The number of input channels.
        depth: The depth of the vision transformer.
        kwargs: Additional keyword arguments for the vision transformer base class.
    """
    def __init__(
        self,
        img_size: int = 1024,  # chosen to match our experiments with segment anything
        in_chans: int = 3,
        depth: int = 12,
        **kwargs
    ):
        if not _timm_import_success:
            raise RuntimeError(
                "The vision transformer backend can only be initialized if timm is installed."
                "Please install timm (using conda/mamba) for using https://github.com/facebookresearch/mae/."
                "and then rerun your code"
            )
        super().__init__(img_size=img_size, depth=depth, **kwargs)
        self.img_size = img_size
        self.in_chans = in_chans
        self.depth = depth

    def convert_to_expected_dim(self, inputs_):
        """@private
        """
        inputs_ = inputs_[:, 1:, :]  # removing the class tokens
        # reshape the outputs to desired shape (N x H*W X C -> N x H x W x C)
        rdim = inputs_.shape[1]
        dshape = int(rdim ** 0.5)  # finding the square root of the outputs for obtaining the patch shape
        inputs_ = torch.unflatten(inputs_, 1, (dshape, dshape))
        inputs_ = inputs_.permute(0, 3, 1, 2)
        return inputs_

    def forward_features(self, x):
        """@private
        """
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        # chunks obtained for getting the projections for conjuctions with upsampling blocks
        _chunks = int(self.depth / 4)
        chunks_for_projection = [_chunks - 1, 2*_chunks - 1, 3*_chunks - 1, 4*_chunks - 1]

        list_from_encoder = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in chunks_for_projection:
                list_from_encoder.append(self.convert_to_expected_dim(x))

        x = self.convert_to_expected_dim(x)
        return x, list_from_encoder[:3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the vision transformer to input data.

        Args:
            x: The input data.

        Returns:
            The vision transformer output.
        """
        x, list_from_encoder = self.forward_features(x)
        return x, list_from_encoder


def get_vision_transformer(backbone: str, model: str, img_size: int = 1024) -> nn.Module:
    """Get vision transformer encoder.

    Args:
        backbone: The name of the vision transformer implementation. One of "sam" or "mae".
        model: The name of the model. One of "vit_b", "vit_l" or "vit_h".
        img_size: The size of the input for the image encoder. Input images will be resized to match this size.

    Returns:
        The vision transformer.
    """
    if backbone == "sam":
        if model == "vit_b":
            encoder = ViT_Sam(
                depth=12, embed_dim=768, img_size=1024, mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=12, patch_size=16, qkv_bias=True, use_rel_pos=True,
                global_attn_indexes=[2, 5, 8, 11],
                window_size=14, out_chans=256,
            )
        elif model == "vit_l":
            encoder = ViT_Sam(
                depth=24, embed_dim=1024, img_size=1024, mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=16, patch_size=16, qkv_bias=True, use_rel_pos=True,
                global_attn_indexes=[5, 11, 17, 23],
                window_size=14,  out_chans=256
            )
        elif model == "vit_h":
            encoder = ViT_Sam(
                depth=32, embed_dim=1280, img_size=1024, mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=16, patch_size=16, qkv_bias=True, use_rel_pos=True,
                global_attn_indexes=[7, 15, 23, 31],
                window_size=14, out_chans=256
            )
        else:
            raise ValueError(f"{model} is not supported by SAM. Currently vit_b, vit_l, vit_h are supported.")

    elif backbone == "mae":
        if model == "vit_b":
            encoder = ViT_MAE(
                img_size=img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
        elif model == "vit_l":
            encoder = ViT_MAE(
                img_size=img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
        elif model == "vit_h":
            encoder = ViT_MAE(
                img_size=img_size, patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
        else:
            raise ValueError(f"{model} is not supported by MAE. Currently vit_b, vit_l, vit_h are supported.")

    else:
        raise ValueError("The UNETR supported backbones are `sam` or `mae`. Please choose either of the two.")

    return encoder
