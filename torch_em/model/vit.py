from functools import partial
from typing import Tuple, List

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

try:
    from sam2.modeling.backbones.hieradet import Hiera
    from sam2.modeling.position_encoding import PositionEmbeddingSine
    from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
    _sam2_import_success = True
except ImportError:
    ImageEncoder = object
    _sam2_import_success = False

try:
    from dinov2.models.vision_transformer import DinoVisionTransformer as DinoV2VisionTransformer
    from dinov2.layers import MemEffAttention, NestedTensorBlock as Block
    _dinov2_import_success = True
except ImportError:
    DinoV2VisionTransformer = object
    _dinov2_import_success = False

try:
    from dinov3.models.vision_transformer import DinoVisionTransformer as DinoV3VisionTransformer
    _dinov3_import_success = True
except ImportError:
    DinoV3VisionTransformer = object
    _dinov3_import_success = False


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
        global_attn_indexes: Tuple[int, ...] = [2, 5, 8, 11],
        **kwargs,
    ) -> None:
        if not _sam_import_success:
            raise RuntimeError(
                "The vision transformer backend can only be initialized if segment anything is installed. "
                "Please install segment anything from https://github.com/facebookresearch/segment-anything "
                "and then rerun your code."
            )

        super().__init__(embed_dim=embed_dim, global_attn_indexes=global_attn_indexes, **kwargs)
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
                "The vision transformer backend can only be initialized if timm is installed. "
                "Please install timm (using conda/mamba) for using https://github.com/facebookresearch/mae/ "
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


class ViT_Sam2(ImageEncoder):
    """Vision Transformer derived from the Segment Anything 2 Codebase (https://arxiv.org/abs/2408.00714).

    Based on https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/backbones/image_encoder.py.

    Args:
        backbone_channel_list: The channels throughout the entire backbone.
        embed_dim: The initial embedding dimension.
        num_heads: The initial number of heads.
        stages: The number of blocks per stage.
        global_att_blocks: The parameter to decide which blocks have global attention.
        window_pos_embed_bkg_spatial_size: The spatial size of windowed positional embedding.
        window_spec: The window size per stage, when not using global attention.
        scalp: The count of lowest resolution features to discard.
    """
    def __init__(
        self,
        backbone_channel_list: List[int],
        img_size: int = 1024,
        embed_dim: int = 96,
        num_heads: int = 1,
        stages: Tuple[int, ...] = (2, 3, 16, 3),
        global_att_blocks: Tuple[int, ...] = (12, 16, 20),
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        window_spec: Tuple[int, ...] = (8, 4, 14, 7),
        scalp: int = 1,
    ):
        if not _sam2_import_success:
            raise RuntimeError(
                "The vision transformer backend can only be initialized if segment anything 2 is installed. "
                "Please install segment anything 2 from https://github.com/facebookresearch/sam2 "
                "and then rerun your code"
            )

        trunk = Hiera(
            embed_dim=embed_dim,
            num_heads=num_heads,
            stages=stages,
            global_att_blocks=global_att_blocks,
            window_pos_embed_bkg_spatial_size=window_pos_embed_bkg_spatial_size,
            window_spec=window_spec,
        )
        neck = FpnNeck(
            position_encoding=PositionEmbeddingSine(num_pos_feats=256),
            d_model=256,
            backbone_channel_list=backbone_channel_list,
            fpn_top_down_levels=[2, 3],
            fpn_interp_model="nearest",
        )

        super().__init__(trunk=trunk, neck=neck, scalp=scalp)
        self.scalp = scalp
        self.embed_dim = embed_dim
        self.img_size = img_size

    def forward(self, x: torch.Tensor):
        # The forward pass throught the backbone.
        features, pos = self.neck(self.trunk(x))
        if self.scalp > 0:  # This discard the "n" lowest resolution features.
            features, pos = features[:-self.scalp], pos[:-self.scalp]

        return features[-1], features


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

    NOTE: For downstream tasks, the "base_resoulution" parameter needs to be adjusted manually when using
    the model on a different zoom factor dataset.
    """

    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=12, base_resolution=2.5, **kwargs
    ):
        super().__init__(img_size=img_size, embed_dim=embed_dim, **kwargs)
        self.img_size = img_size
        self.in_chans = in_chans
        self.depth = depth
        self.base_resolution = base_resolution

        self.patch_embed = PatchEmbedUnSafe(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

    def transform_inputs(self, x):
        import kornia.augmentation as K
        from kornia.constants import Resample

        self._transforms = CustomCompose(
            rescale_transform=K.RandomResizedCrop(
                (448, 448),
                ratio=(1.0, 1.0),
                scale=(1.0, 1.0),
                resample=Resample.BICUBIC.name,
            ),
            other_transforms=None,
            src_transform=K.Resize((224, 224)),
        )
        x, _, ratios, _, _ = self._transforms(x)
        input_res = ratios * self.base_resolution
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

        # chunks obtained for getting the projections for conjuctions with upsampling blocks
        _chunks = int(self.depth / 4)
        chunks_for_projection = [_chunks - 1, 2*_chunks - 1, 3*_chunks - 1, 4*_chunks - 1]

        list_from_encoder = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in chunks_for_projection:
                list_from_encoder.append(self.convert_to_expected_dim(x))

        x = self.convert_to_expected_dim(x)

        return x, list_from_encoder

    def forward(self, x):
        x, list_from_encoder = self.forward_features(x)
        return x, list_from_encoder


class ViT_DINOv2(DinoV2VisionTransformer):
    """Vision Transformer derived from the DINOv2 Codebase (https://arxiv.org/abs/2304.07193).

    Based on:
    https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        depth: int = 12,
        num_register_tokens: int = 0,
        **kwargs
    ):
        if not _dinov2_import_success:
            raise RuntimeError(
                "The vision transformer backend can only be initialized if DINOv2 is installed. "
                "Please install DINOv2 from https://github.com/facebookresearch/dinov2 "
                "and then rerun your code."
            )

        super().__init__(
            img_size=img_size,
            depth=depth,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
            **kwargs
        )

        self.img_size = img_size
        self.num_register_tokens = num_register_tokens
        self.patch_size = patch_size
        self.attn_outs = [i for i in range(depth) if i % 3 == 2]

    def forward(self, x, masks=None) -> torch.Tensor:

        B = x.shape[0]

        x = self.prepare_tokens_with_masks(x)

        list_of_encoder = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.attn_outs:
                list_of_encoder.append(x)

        x = self.norm(x)
        x = x[:, self.num_register_tokens + 1:].reshape(
            B, self.img_size // self.patch_size, self.img_size // self.patch_size, -1
        ).permute(0, 3, 1, 2).contiguous()

        list_of_encoder = [
            o[:, self.num_register_tokens + 1:].reshape(
                B, self.img_size // self.patch_size, self.img_size // self.patch_size, -1
            ).permute(0, 3, 1, 2).contiguous() for o in list_of_encoder
        ]

        return x, list_of_encoder[:3]


class ViT_DINOv3(DinoV3VisionTransformer):
    """Vision Transformer derived from the DINOv3 Codebase (https://arxiv.org/abs/2508.10104).

    Based on:
    https://github.com/facebookresearch/dinov3/blob/main/dinov3/models/vision_transformer.py.

    Args:
        img_size: The input image size.
        patch_size: The patch size.
        embed_dim: The embedding dimension.
        depth: The depth of the network.
        num_heads: The number of heads.
        ffn_ratio: The FFN rato.
        n_storage_tokens: The number of storage (class) tokens to remove.
        kwargs: Keyword arguments for the image encoder base class.
    """
    def __init__(
        self,
        in_chans: int = 3,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        n_storage_tokens: int = 0,
        **kwargs
    ):
        if not _dinov3_import_success:
            raise RuntimeError(
                "The vision transformer backend can only be initialized if DINOv3 is installed. "
                "Please install DINOv3 from https://github.com/facebookresearch/dinov3 "
                "and then rerun your code."
            )

        super().__init__(
            in_chans=in_chans,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            n_storage_tokens=n_storage_tokens,
            **kwargs
        )

        self.in_chans = in_chans
        self.img_size = img_size
        self.n_storage_tokens = n_storage_tokens
        self.attn_outs = [i for i in range(depth) if i % 3 == 2]

    def forward(self, x) -> torch.Tensor:

        B = x.shape[0]

        x, hw_tuple = self.prepare_tokens_with_masks(x)

        list_of_encoder = []
        for i, blk in enumerate(self.blocks):
            rope_sincos = self.rope_embed(H=hw_tuple[0], W=hw_tuple[1])
            x = blk(x, rope_sincos)
            if i in self.attn_outs:
                list_of_encoder.append(x)

        x = self.norm(x)
        x = x[:, self.n_storage_tokens + 1:].reshape(
            B, self.img_size // self.patch_size, self.img_size // self.patch_size, -1
        ).permute(0, 3, 1, 2).contiguous()

        list_of_encoder = [
            o[:, self.n_storage_tokens + 1:].reshape(
                B, self.img_size // self.patch_size, self.img_size // self.patch_size, -1
            ).permute(0, 3, 1, 2).contiguous() for o in list_of_encoder
        ]

        return x, list_of_encoder[:3]


def get_vision_transformer(backbone: str, model: str, img_size: int = 1024, **kwargs) -> nn.Module:
    """Get vision transformer encoder.

    Args:
        backbone: The name of the vision transformer implementation. One of "sam" / "mae" / "scalemae".
        model: The name of the model. One of "vit_b", "vit_l" or "vit_h".
        img_size: The size of the input for the image encoder. Input images will be resized to match this size.
        kwargs: Additional kwargs which can be expected by the vision transformer,
            e.g. 'base_resolution' for `ViT_ScaleMAE`.

    Returns:
        The vision transformer.
    """
    if backbone == "sam":
        if model == "vit_b":
            encoder = ViT_Sam(
                depth=12, embed_dim=768, img_size=img_size, mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=12, patch_size=16, qkv_bias=True, use_rel_pos=True,
                global_attn_indexes=[2, 5, 8, 11],
                window_size=14, out_chans=256,
            )
        elif model == "vit_l":
            encoder = ViT_Sam(
                depth=24, embed_dim=1024, img_size=img_size, mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=16, patch_size=16, qkv_bias=True, use_rel_pos=True,
                global_attn_indexes=[5, 11, 17, 23],
                window_size=14, out_chans=256,
            )
        elif model == "vit_h":
            encoder = ViT_Sam(
                depth=32, embed_dim=1280, img_size=img_size, mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=16, patch_size=16, qkv_bias=True, use_rel_pos=True,
                global_attn_indexes=[7, 15, 23, 31],
                window_size=14, out_chans=256,
            )
        else:
            raise ValueError(f"'{model}' is not supported by SAM. Currently, 'vit_b', 'vit_l', 'vit_h' are supported.")

    elif backbone == "sam2":
        if model == "hvit_t":
            encoder = ViT_Sam2(
                img_size=img_size, embed_dim=96, num_heads=1, stages=[1, 2, 7, 2], global_att_blocks=[5, 7, 9],
                window_pos_embed_bkg_spatial_size=[7, 7], backbone_channel_list=[768, 384, 192, 96],
            )
        elif model == "hvit_s":
            encoder = ViT_Sam2(
                img_size=img_size, embed_dim=96, num_heads=1, stages=[1, 2, 11, 2], global_att_blocks=[7, 10, 13],
                window_pos_embed_bkg_spatial_size=[7, 7], backbone_channel_list=[768, 384, 192, 96],
            )
        elif model == "hvit_b":
            encoder = ViT_Sam2(
                img_size=img_size, embed_dim=112, num_heads=2, backbone_channel_list=[896, 448, 224, 112],
            )
        elif model == "hvit_l":
            encoder = ViT_Sam2(
                img_size=img_size, embed_dim=144, num_heads=2, stages=[2, 6, 36, 4], global_att_blocks=[23, 33, 43],
                window_spec=[8, 4, 16, 8], backbone_channel_list=[1152, 576, 288, 144],
            )
        else:
            raise ValueError(
                f"'{model}' is not supported by SAM2. Currently, 'hvit_t', 'hvit_s', 'hvit_b', 'hvit_l' are supported."
            )

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
        base_resolution = kwargs.get("base_resolution", 2.5)

        if model == "vit_b":
            encoder = ViT_ScaleMAE(
                img_size=img_size, patch_size=8, embed_dim=768, depth=12, num_heads=12,
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                base_resolution=base_resolution,
            )
        elif model == "vit_l":
            encoder = ViT_ScaleMAE(
                img_size=img_size, patch_size=8, embed_dim=1024, depth=24, num_heads=16,
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                base_resolution=base_resolution,
            )
        elif model == "vit_h":
            encoder = ViT_ScaleMAE(
                img_size=img_size, patch_size=8, embed_dim=1280, depth=32, num_heads=16,
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                base_resolution=base_resolution,
            )
        else:
            raise ValueError(
                f"'{model}' is not supported by ScaleMAE. Currently, 'vit_b', 'vit_l' and 'vit_h' are supported."
            )

    elif backbone == "dinov2":
        block_fn = partial(Block, attn_class=MemEffAttention)
        msg = "The model name should be either 'vit_<X>' or 'vit_<X>_reg<Y>."

        if model.startswith("vit_s"):
            assert model in ["vit_s", "vit_s_reg4"], msg
            encoder = ViT_DINOv2(
                img_size=img_size, patch_size=14, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
                block_fn=block_fn, in_chans=3, channel_adaptive=False, init_values=1e-5, block_chunks=0,
                num_register_tokens=4 if model.endswith("_reg4") else 0,
            )
        elif model.startswith("vit_b"):
            assert model in ["vit_b", "vit_b_reg4"], msg
            encoder = ViT_DINOv2(
                img_size=img_size, patch_size=14, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                block_fn=block_fn, in_chans=3, channel_adaptive=False, init_values=1e-5, block_chunks=0,
                num_register_tokens=4 if model.endswith("_reg4") else 0,
            )
        elif model.startswith("vit_l"):
            assert model in ["vit_l", "vit_l_reg4"], msg
            encoder = ViT_DINOv2(
                img_size=img_size, patch_size=14, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
                block_fn=block_fn, in_chans=3, channel_adaptive=False, init_values=1e-5, block_chunks=0,
                num_register_tokens=4 if model.endswith("_reg4") else 0,
            )
        elif model.startswith("vit_g"):
            assert model in ["vit_g", "vit_g_reg4"], msg
            encoder = ViT_DINOv2(
                img_size=img_size, patch_size=14, embed_dim=1536, depth=40, num_heads=24, mlp_ratio=4,
                block_fn=block_fn, in_chans=3, channel_adaptive=False, init_values=1e-5, block_chunks=0,
                num_register_tokens=4 if model.endswith("_reg4") else 0, ffn_layer="swiglu",
            )
        else:
            raise ValueError(
                f"'{model}' is not supported by DINOv2. Currently, 'vit_s', 'vit_b', 'vit_l' and 'vit_g' are supported."
            )

    elif backbone == "dinov3":

        if model == "vit_s":
            encoder = ViT_DINOv3(
                img_size=img_size, pos_embed_rope_rescale_coords=2, pos_embed_rope_dtype="fp32", embed_dim=384,
                num_heads=6, layerscale_init=1.0e-05, norm_layer="layernormbf16", n_storage_tokens=4, mask_k_bias=True,
            )
        elif model == "vit_s+":
            encoder = ViT_DINOv3(
                img_size=img_size, pos_embed_rope_rescale_coords=2, pos_embed_rope_dtype="fp32", embed_dim=384,
                num_heads=6, ffn_ratio=6, layerscale_init=1.0e-05, norm_layer="layernormbf16",
                ffn_layer="swiglu", n_storage_tokens=4, mask_k_bias=True,
            )

        elif model == "vit_b":
            encoder = ViT_DINOv3(
                img_size=img_size, pos_embed_rope_rescale_coords=2, pos_embed_rope_dtype="fp32",
                layerscale_init=1.0e-05, norm_layer="layernormbf16", n_storage_tokens=4, mask_k_bias=True,
            )
        elif model == "vit_l":
            encoder = ViT_DINOv3(
                img_size=img_size, pos_embed_rope_rescale_coords=2, pos_embed_rope_dtype="fp32", embed_dim=1024,
                depth=24, num_heads=16, layerscale_init=1.0e-05, norm_layer="layernormbf16",
                n_storage_tokens=4, mask_k_bias=True,
            )
        elif model == "vit_l+":
            encoder = ViT_DINOv3(
                img_size=img_size, pos_embed_rope_rescale_coords=2, pos_embed_rope_dtype="fp32", embed_dim=1024,
                depth=24, num_heads=16, ffn_ratio=6.0, layerscale_init=1.0e-05, norm_layer="layernormbf16",
                ffn_layer="swiglu", n_storage_tokens=4, mask_k_bias=True,
            )
        elif model == "vit_h+":
            encoder = ViT_DINOv3(
                img_size=img_size, pos_embed_rope_rescale_coords=2, pos_embed_rope_dtype="fp32", embed_dim=1280,
                depth=32, num_heads=20, ffn_ratio=6.0, layerscale_init=1.0e-05, norm_layer="layernormbf16",
                ffn_layer="swiglu", n_storage_tokens=4, mask_k_bias=True,
            )
        elif model == "vit_7b":
            encoder = ViT_DINOv3(
                img_size=img_size, pos_embed_rope_rescale_coords=2, pos_embed_rope_dtype="fp32", embed_dim=4096,
                depth=40, num_heads=32, ffn_ratio=3, qkv_bias=False, drop_path_rate=0.0, layerscale_init=1.0e-05,
                norm_layer="layernormbf16", ffn_layer="swiglu64", n_storage_tokens=4, mask_k_bias=True,
                untie_global_and_local_cls_norm=True,
            )
        else:
            raise ValueError(
                f"'{model}' is not supported by DINOv3. Currently, "
                " 'vit_s', 'vit_s+', 'vit_b', 'vit_l', 'vit_l+', 'vit_h+'. 'vit_7b' are supported."
            )

    else:
        raise ValueError(
            "The 'UNETR' supported backbones are 'sam', 'sam2', 'mae', 'scalemae' or 'dinov3'. "
            "Please choose one of them."
        )

    return encoder
