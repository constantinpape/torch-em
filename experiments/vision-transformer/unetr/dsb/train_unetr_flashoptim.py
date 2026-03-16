import os

import torch

import torch_em
from torch_em.model.unetr import UNETR
from torch_em.trainer.flashoptim_trainer import FlashOptimTrainer

from micro_sam.util import _download_sam_model

from common import get_loaders


def setup_env():
    """Prepends the linker and CUDA include paths required for FlashOptim kernel compilation.

    The CUDA include path is derived from the version PyTorch was built against,
    with a fallback to the `/usr/local/cuda` classic symlink if the versioned directory is absent.

    NOTE: This is currently optimized for NHR cluster users.
    """
    cuda_version = torch.version.cuda  # e.g. "12.8"
    cuda_include = f"/usr/local/cuda-{cuda_version}/targets/x86_64-linux/include"
    if not os.path.isdir(cuda_include):
        cuda_include = "/usr/local/cuda/targets/x86_64-linux/include"
    os.environ["LIBRARY_PATH"] = "/usr/lib64:" + os.environ.get("LIBRARY_PATH", "")
    os.environ["CPATH"] = cuda_include + ":" + os.environ.get("CPATH", "")


def get_model(pretrained):
    model_type = "vit_b"
    checkpoint = _download_sam_model(model_type=model_type)[0] if pretrained else None
    model = UNETR(
        backbone="sam",
        encoder=model_type,
        out_channels=3,
        encoder_checkpoint_path=checkpoint,
        use_sam_stats=pretrained,
        final_activation="Sigmoid",
        use_skip_connection=False,
    )
    return model


def train_unetr(pretrained, use_dice, mask_background):
    model = get_model(pretrained)

    if use_dice:
        loss = torch_em.loss.DiceBasedDistanceLoss(mask_distances_in_bg=mask_background)
        name = "distance_unetr-dice-flashoptim"
    else:
        loss = torch_em.loss.DistanceLoss(mask_distances_in_bg=mask_background)
        name = "distance_unetr-dist-loss-flashoptim"

    if mask_background:
        name += "-mask-bg"

    if pretrained:
        name += "-pretrained"

    train_loader, val_loader = get_loaders(True)

    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=loss,
        learning_rate=1e-4,
        device=torch.device("cuda"),
        log_image_interval=100,
        trainer_class=FlashOptimTrainer,
    )
    trainer.fit(iterations=int(1e4))


if __name__ == "__main__":
    setup_env()
    train_unetr(pretrained=True, use_dice=True, mask_background=True)
