"""Shared configuration for vision transformer models and raw transforms."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ViTConfig:
    """Defaults shared by ViT construction, normalization, and UNETR."""

    models: Tuple[str, ...]
    img_size: int
    stats: Optional[str]
    dependency: str
    clip_max: int = 1
    skip_models: Tuple[str, ...] = ()
    weight_suffixes: bool = False


VIT_CONFIGS = {
    "sam": ViTConfig(
        models=("vit_b", "vit_l", "vit_h"), img_size=1024, stats="sam", dependency="sam", clip_max=255,
        skip_models=("vit_b", "vit_l", "vit_h"), weight_suffixes=True
    ),
    "sam2": ViTConfig(
        models=("hvit_t", "hvit_s", "hvit_b", "hvit_l"), img_size=1024, stats="sam", dependency="sam2",
        weight_suffixes=True
    ),
    "sam3": ViTConfig(models=("vit_pe",), img_size=1008, stats="sam", dependency="sam3", weight_suffixes=True),
    "cellpose_sam": ViTConfig(models=("vit_l",), img_size=256, stats=None, dependency="sam"),
    "mae": ViTConfig(
        models=("vit_b", "vit_l", "vit_h"), img_size=224, stats="mae", dependency="timm", skip_models=("vit_b", "vit_l")
    ),
    "scalemae": ViTConfig(models=("vit_b", "vit_l", "vit_h"), img_size=224, stats="mae", dependency="timm"),
    "dinov2": ViTConfig(
        models=("vit_s", "vit_b", "vit_l", "vit_g", "vit_s_reg4", "vit_b_reg4", "vit_l_reg4", "vit_g_reg4"),
        img_size=518, stats="dino", dependency="dinov2"
    ),
    "dinov3": ViTConfig(
        models=("vit_s", "vit_s+", "vit_b", "vit_l", "vit_l+", "vit_h+", "vit_7b"),
        img_size=224, stats="dino", dependency="dinov3",
        skip_models=("vit_s", "vit_s+", "vit_b", "vit_l", "vit_l+", "vit_h+", "vit_7b")
    ),
    "torchvision": ViTConfig(
        models=("vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14"),
        img_size=224, stats="imagenet", dependency="torchvision", skip_models=("vit_b_16", "vit_l_16")
    ),
}
