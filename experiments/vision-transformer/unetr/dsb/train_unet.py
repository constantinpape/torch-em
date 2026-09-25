import torch
import torch_em
from torch_em.model import UNet2d

from common import get_loaders


def train_unet(use_dice, mask_background):
    model = UNet2d(in_channels=1, out_channels=3, initial_features=64, final_activation="Sigmoid")

    n_iterations = 10_000

    if use_dice:
        loss = torch_em.loss.DiceBasedDistanceLoss(mask_distances_in_bg=mask_background)
        name = "distance_unet-dice"
    else:
        loss = torch_em.loss.DistanceLoss(mask_distances_in_bg=mask_background)
        name = "distance_unet-dist-loss"

    if mask_background:
        name += "-mask-bg"

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
        mixed_precision=True,
        log_image_interval=100,
        compile_model=False,
    )
    trainer.fit(n_iterations)


if __name__ == "__main__":
    train_unet(use_dice=False, mask_background=True)
