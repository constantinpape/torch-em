import argparse
import os

import torch
import torch_em
from torch_em.model import UNet2d

OFFSETS = [
    [-1, 0], [0, -1],
    [-3, 0], [0, -3],
    [-9, 0], [0, -9],
    [-27, 0], [0, -27]
]


def train_affinties(root, batch_size, iterations):
    n_out = len(OFFSETS) + 1
    model = UNet2d(in_channels=1, out_channels=n_out, initial_features=64)

    # transform to go from instance segmentation labels
    # to foreground/background and boundary channel
    label_transform = torch_em.transform.AffinityTransform(
        offsets=OFFSETS,
        add_binary_target=True,
        add_mask=True
    )

    patch_shape = (1, 256, 256)
    train_loader = torch_em.default_segmentation_loader(
        os.path.join(root, "train/images"), "*.tif",
        os.path.join(root, "train/masks"), "*.tif",
        batch_size=batch_size, patch_shape=patch_shape,
        label_transform=label_transform
    )
    val_loader = torch_em.default_segmentation_loader(
        os.path.join(root, "test/images"), "*.tif",
        os.path.join(root, "test/masks"), "*.tif",
        batch_size=batch_size, patch_shape=patch_shape,
        label_transform=label_transform
    )

    loss = torch_em.loss.LossWrapper(
        torch_em.loss.DiceLoss(),
        transform=torch_em.loss.ApplyAndRemoveMask()
    )

    # the trainer object that handles the training details
    # the model checkpoints will be saved in "checkpoints/dsb-boundary-model"
    # the tensorboard logs will be saved in "logs/dsb-boundary-model"
    trainer = torch_em.default_segmentation_trainer(
        name="dsb-affinity-model",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=loss,
        learning_rate=1e-4,
        device=torch.device("cuda"),
        mixed_precision=True,
        log_image_interval=50
    )
    trainer.fit(iterations=iterations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str,
                        help="Path to dsb2018 folder with train and test subfolders")
    parser.add_argument('--batch_size', '-b', type=int, default=8)
    parser.add_argument('--iterations', '-i', type=int, default=int(1e4))
    args = parser.parse_args()
    train_affinties(args.root, args.batch_size, args.iterations)
