from functools import partial

import numpy as np
import torch_em
from torch_em.model import UNet2d
from torch_em.data.datasets import get_isbi_loader
from torch_em.util import parser_helper


def get_model():
    n_out = 1
    model = UNet2d(
        in_channels=1,
        out_channels=n_out,
        final_activation="Sigmoid"
    )
    return model


def train_boundaries(input_path, n_iterations, device):
    model = get_model()

    batch_size = 1
    # shape of input patches (blocks) used for training
    patch_shape = [1, 512, 512]

    normalization = partial(
        torch_em.transform.raw.normalize,
        minval=0, maxval=255
    )

    roi_train = np.s_[:28, :, :]
    train_loader = get_isbi_loader(
        input_path,
        download=True,
        boundaries=True,
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
        boundaries=True,
        patch_shape=patch_shape,
        rois=roi_val,
        batch_size=batch_size,
        raw_transform=normalization,
        num_workers=8*batch_size,
        shuffle=True
    )

    loss = torch_em.loss.DiceLoss()

    name = "boundary-model"
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


if __name__ == "__main__":
    parser = parser_helper()
    args = parser.parse_args()
    train_boundaries(args.input, args.n_iterations, args.device)
