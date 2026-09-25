from functools import partial

import numpy as np
import torch_em
from torch_em.model import UNet2d
from torch_em.data.datasets import get_isbi_loader
from torch_em.util import parser_helper

OFFSETS = [
    [-1, 0], [0, -1],
    [-3, 0], [0, -3],
    [-9, 0], [0, -9],
    [-27, 0], [0, -27]
]

DIAG_OFFSETS = [[-2, -9],
                [-5, -7],
                [-7, -5],
                [-9, -2],
                [-9, 2],
                [-7, 5],
                [-5, 7],
                [-2, 9]]


def get_offsets(use_diagonal_offsets):
    if use_diagonal_offsets:
        offsets = OFFSETS[:4] + DIAG_OFFSETS + OFFSETS[6:]
    else:
        offsets = OFFSETS
    return offsets


def get_model(use_diagonal_offsets):
    n_out = len(get_offsets(use_diagonal_offsets))
    model = UNet2d(
        in_channels=1,
        out_channels=n_out,
        final_activation="Sigmoid"
    )
    return model


def train_affinities(input_path, n_iterations, device, use_diagonal_offsets):

    model = get_model(use_diagonal_offsets)
    offsets = get_offsets(use_diagonal_offsets)

    # shape of input patches (blocks) used for training
    patch_shape = [1, 512, 512]
    batch_size = 1

    normalization = partial(
        torch_em.transform.raw.normalize,
        minval=0, maxval=255
    )

    roi_train = np.s_[:28, :, :]
    train_loader = get_isbi_loader(
        input_path,
        download=True,
        offsets=offsets,
        patch_shape=patch_shape,
        rois=roi_train,
        batch_size=batch_size,
        raw_transform=normalization,
        num_workers=8*batch_size,
        shuffle=True
    )

    roi_val = np.s_[28:, :, :]
    val_loader = get_isbi_loader(
        input_path,
        download=False,
        offsets=offsets,
        patch_shape=patch_shape,
        rois=roi_val,
        batch_size=batch_size,
        raw_transform=normalization,
        num_workers=8*batch_size,
        shuffle=True
    )

    loss = torch_em.loss.LossWrapper(
        torch_em.loss.DiceLoss(),
        transform=torch_em.loss.ApplyAndRemoveMask()
    )

    name = "affinity-model"
    if use_diagonal_offsets:
        name += "_diagonal_offsets"
    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=loss,
        learning_rate=1e-4,
        mixed_precision=True,
        log_image_interval=50,
        device=device
    )
    trainer.fit(n_iterations)


def print_the_offsets(use_diagonal_offsets):
    offs = get_offsets(use_diagonal_offsets)
    print(offs)


if __name__ == "__main__":
    parser = parser_helper()
    parser.add_argument("-d", "--use_diagonal_offsets", type=int, default=0)
    parser.add_argument("-p", "--print_offsets", default=0)
    args = parser.parse_args()
    if bool(args.print_offsets):
        print_the_offsets(bool(args.use_diagonal_offsets))
    else:
        train_affinities(args.input, args.n_iterations, args.device, bool(args.use_diagonal_offsets))
