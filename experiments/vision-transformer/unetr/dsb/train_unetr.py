import torch
import torch_em
from torch_em.model.unetr import UNETR

from common import get_loaders


def get_model(pretrained):
    checkpoint = "/home/nimcpape/.cache/micro_sam/models/vit_b" if pretrained else None
    model = UNETR(
        backbone="sam", encoder="vit_b", out_channels=3,
        encoder_checkpoint_path=checkpoint,
        use_sam_stats=pretrained, final_activation="Sigmoid",
    )
    return model


def train_unetr(pretrained, use_dice, mask_background):
    model = get_model(pretrained)

    n_iterations = 10_000

    if use_dice:
        loss = torch_em.loss.DiceBasedDistanceLoss(mask_distances_in_bg=mask_background)
        name = "distance_unetr-dice"
    else:
        loss = torch_em.loss.DistanceLoss(mask_distances_in_bg=mask_background)
        name = "distance_unetr-dist-loss"

    if mask_background:
        name += "-mask-bg"

    if pretrained:
        name += "-pretrained"

    train_loader, val_loader = get_loaders(True)

    # the trainer object that handles the training details
    # the model checkpoints will be saved in "checkpoints/dsb-boundary-model"
    # the tensorboard logs will be saved in "logs/dsb-boundary-model"
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
    train_unetr(pretrained=True, use_dice=False, mask_background=True)
