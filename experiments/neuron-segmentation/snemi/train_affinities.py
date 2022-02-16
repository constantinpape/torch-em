import numpy as np
import torch_em
from torch_em.model import AnisotropicUNet
from torch_em.loss import DiceLoss, LossWrapper, ApplyAndRemoveMask
from torch_em.data.datasets import get_snemi_loader
from torch_em.util import parser_helper

OFFSETS = [
    [-1, 0, 0], [0, -1, 0], [0, 0, -1],
    [-2, 0, 0], [0, -3, 0], [0, 0, -3],
    [-3, 0, 0], [0, -9, 0], [0, 0, -9],
    [-4, 0, 0], [0, -27, 0], [0, 0, -27]
]


def get_loader(input_path, train, patch_shape, batch_size=1, n_samples=None):
    n_slices = 100
    z = n_slices - patch_shape[0]
    roi = np.s_[:z, :, :] if train else np.s_[z:, :, :]
    return get_snemi_loader(
        path=input_path,
        patch_shape=patch_shape,
        batch_size=batch_size,
        rois=roi,
        offsets=OFFSETS,
        n_samples=n_samples,
        num_workers=8*batch_size,
        shuffle=True,
        download=True
    )


def get_model():
    n_out = len(OFFSETS)
    model = AnisotropicUNet(
        scale_factors=[
            [1, 2, 2],
            [1, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2]
        ],
        in_channels=1,
        out_channels=n_out,
        initial_features=32,
        gain=2,
        final_activation="Sigmoid"
    )
    return model


def train_affinities(args):
    model = get_model()
    patch_shape = [32, 320, 320]

    train_loader = get_loader(
        args.input, train=True,
        patch_shape=patch_shape,
        n_samples=1000
    )
    val_loader = get_loader(
        args.input, train=False,
        patch_shape=patch_shape,
        n_samples=50
    )

    loss = LossWrapper(loss=DiceLoss(), transform=ApplyAndRemoveMask())
    name = "affinity_model"
    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=loss,
        learning_rate=1e-4,
        mixed_precision=True,
        log_image_interval=50
    )

    if args.from_checkpoint:
        trainer.fit(args.n_iterations, "latest")
    else:
        trainer.fit(args.n_iterations)


def check(args, train=True, val=True, n_images=2):
    from torch_em.util.debug import check_loader
    patch_shape = [32, 320, 320]
    if train:
        print("Check train loader")
        loader = get_loader(args.input, True, patch_shape)
        check_loader(loader, n_images)
    if val:
        print("Check val loader")
        loader = get_loader(args.input, False, patch_shape)
        check_loader(loader, n_images)


if __name__ == "__main__":
    parser = parser_helper()
    args = parser.parse_args()
    if args.check:
        check(args, train=True, val=True)
    else:
        train_affinities(args)
