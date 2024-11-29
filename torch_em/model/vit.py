from typing import Tuple
from functools import partial

import torch
import torch.nn as nn

# we catch ImportErrors here because segment_anything, micro_sam, scale_mae and timm should
# only be optional dependencies for torch_em
try:
    from segment_anything.modeling import ImageEncoderViT
    _sam_import_success = True
except ImportError:
    ImageEncoderViT = object
    _sam_import_success = False

try:
    from timm.models.vision_transformer import VisionTransformer, PatchEmbed
    _timm_import_success = True
except ImportError:
    VisionTransformer = object
    PatchEmbed = object
    _timm_import_success = False


class ViT_Sam(ImageEncoderViT):
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

        super().__init__(embed_dim=embed_dim, global_attn_indexes=global_attn_indexes, **kwargs)
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
        return x, list_from_encoder[:3]


class ViT_MAE(VisionTransformer):
    """Vision Transformer derived from the Masked Auto Encoder Codebase (https://arxiv.org/abs/2111.06377)
    https://github.com/facebookresearch/mae/blob/main/models_vit.py#L20-L53
    """
    def __init__(
        self,
        img_size=1024,  # chosen to match our experiments with Segment Anything
        in_chans=3,
        depth=12,
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
        inputs_ = inputs_[:, 1:, :]  # removing the class tokens
        # reshape the outputs to desired shape (N x H*W X C -> N x H x W x C)
        rdim = inputs_.shape[1]
        dshape = int(rdim ** 0.5)  # finding the square root of the outputs for obtaining the patch shape
        inputs_ = torch.unflatten(inputs_, 1, (dshape, dshape))
        inputs_ = inputs_.permute(0, 3, 1, 2)
        return inputs_

    def forward_features(self, x):
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

    def forward(self, x):
        x, list_from_encoder = self.forward_features(x)
        return x, list_from_encoder


#
# Utilities for ScaleMAE's ViT
#


class CustomCompose:
    def __init__(self, rescale_transform, other_transforms, src_transform):
        self.rescale_transform = rescale_transform
        self.other_transforms = other_transforms
        self.src_transform = src_transform

    def __call__(self, x, valid_masks=None):
        if valid_masks is not None:
            nodata = (x * (1 - valid_masks.float())).max()
        x_aug = self.rescale_transform(x)
        parms = self.rescale_transform._params

        # sanity check, comment if this is working
        # valid_masks = self.rescale_transform(valid_masks.float(), params=parms)
        # assert (x_aug==self.rescale_transform(x, params=parms)).all() #

        if valid_masks is not None:
            valid_masks = x_aug != nodata
            _, c, h, w = x_aug.shape
            zero_ratio = ((valid_masks == 0).sum((1, 2, 3)) / (h * w * c)).cpu().numpy()
        else:
            zero_ratio = -1

        if self.other_transforms:
            x_aug = self.other_transforms(x_aug)
        x_src = self.src_transform(x_aug)
        dx = parms["src"][:, 1, 0] - parms["src"][:, 0, 0]

        # dy = (parms['src'][:,2,1] - parms['src'][:,1,1])
        # assert (dx == dy).all()

        h, w = x_aug.shape[-2:]
        # assert h == w

        return x_aug, x_src, dx / h, zero_ratio, valid_masks


def get_2d_sincos_pos_embed_with_resolution(embed_dim, grid_size, res, cls_token=False, device="cpu"):
    """
    grid_size: int of the grid height and width
    res: array of size n, representing the resolution of a pixel (say, in meters),
    return:
    pos_embed: [n,grid_size*grid_size, embed_dim] or [n,1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    # res = torch.FloatTensor(res).to(device)
    res = res.to(device)
    grid_h = torch.arange(grid_size, dtype=torch.float32, device=device)
    grid_w = torch.arange(grid_size, dtype=torch.float32, device=device)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")  # here h goes first,direction reversed for numpy
    grid = torch.stack(grid, dim=0)  # 2 x h x w

    # grid = grid.reshape([2, 1, grid_size, grid_size])
    grid = torch.einsum("chw,n->cnhw", grid, res)  # 2 x n x h x w
    _, n, h, w = grid.shape
    pos_embed = get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid)  # (nxH*W, D/2)
    pos_embed = pos_embed.reshape(n, h * w, embed_dim)
    if cls_token:
        pos_embed = torch.cat(
            [torch.zeros([n, 1, embed_dim], dtype=torch.float32, device=pos_embed.device), pos_embed], dim=1
        )

    return pos_embed


def get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    # old_shape = pos
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


class PatchEmbedUnSafe(PatchEmbed):
    """Image to Patch Embedding"""

    def forward(self, x):
        B, C, H, W = x.shape

        # NOTE: Comment code from ScaleMAE: Dropped size check in timm
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ViT_ScaleMAE(VisionTransformer):
    """Vision Transformer dervied from the Scale Masked Auto Encoder codebase (TODO: paper and github link).
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, **kwargs):
        super().__init__(img_size=img_size, embed_dim=embed_dim, **kwargs)
        self.img_size = img_size
        self.in_chans = in_chans

        self.patch_embed = PatchEmbedUnSafe(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

    def transform_inputs(self, x):
        import kornia.augmentation as K
        from kornia.constants import Resample

        # "base_resoulution" needs to be adjusted manually when using the model on a different zoom factor dataset.
        base_resolution = 2.5

        self._transforms = CustomCompose(
            rescale_transform=K.RandomResizedCrop(
                (448, 448),
                ratio=(1.0, 1.0),
                scale=(0.2, 1.0),
                resample=Resample.BICUBIC.name,
            ),
            other_transforms=None,
            src_transform=K.Resize((224, 224)),
        )
        x, _, ratios, _, _ = self._transforms(x)
        input_res = ratios * base_resolution
        return x, input_res

    def convert_to_expected_dim(self, x):
        inputs_ = x[:, 1:, :]  # removing the class tokens
        # reshape the outputs to desired shape (N X H*W X C -> N X H X W X C)
        rdim = inputs_.shape[1]
        dshape = int(rdim ** 0.5)  # finding square root of the outputs for obtaining the patch shape
        inputs_ = torch.unflatten(inputs_, 1, (dshape, dshape))
        inputs_ = inputs_.permute(0, 3, 1, 2)
        return inputs_

    def forward_features(self, x):
        x, input_res = self.transform_inputs(x)

        B, _, h, w = x.shape
        x = self.patch_embed(x)

        num_patches = int((h * w) / (self.patch_embed.patch_size[0] * self.patch_embed.patch_size[1]))
        pos_embed = get_2d_sincos_pos_embed_with_resolution(
            x.shape[-1],
            int(num_patches ** 0.5),
            input_res,
            cls_token=True,
            device=x.device,
        )

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.convert_to_expected_dim(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x


def get_vision_transformer(backbone: str, model: str, img_size: int = 1024):
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
            raise ValueError(f"'{model}' is not supported by SAM. Currently, 'vit_b', 'vit_l', 'vit_h' are supported.")

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
            raise ValueError(f"'{model}' is not supported by MAE. Currently, 'vit_b', 'vit_l', 'vit_h' are supported.")

    elif backbone == "scalemae":
        if model == "vit_b":
            encoder = ViT_ScaleMAE(
                img_size=img_size, patch_size=8, embed_dim=768, depth=12, num_heads=12,
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
        elif model == "vit_l":
            encoder = ViT_ScaleMAE(
                img_size=img_size, patch_size=8, embed_dim=1024, depth=24, num_heads=16,
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
        elif model == "vit_h":
            encoder = ViT_ScaleMAE(
                img_size=img_size, patch_size=8, embed_dim=1280, depth=32, num_heads=16,
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
        else:
            raise ValueError(
                f"'{model}' is not supported by ScaleMAE. Currently, 'vit_b', 'vit_l' and 'vit_h' are supported."
            )

    else:
        raise ValueError("The 'UNETR' supported backbones are `sam`, `mae` or 'scalemae. Please choose one of them.")

    return encoder
