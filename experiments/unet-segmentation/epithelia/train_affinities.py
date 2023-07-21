import os
from glob import glob

import torch_em
from torch_em.model import UNet2d
from torch_em.util import parser_helper

OFFSETS = [
    [-1, 0], [0, -1],
    [-3, 0], [0, -3],
    [-9, 0], [0, -9],
    [-27, 0], [0, -27]
]


def get_model():
    n_out = len(OFFSETS)
    model = UNet2d(
        in_channels=1,
        out_channels=n_out,
        final_activation="Sigmoid"
    )
    return model


def get_loader(args, split, patch_shape):
    paths = glob(os.path.join(args.input, split, "*.zarr"))
    assert len(paths) > 0
    label_transform = torch_em.transform.AffinityTransform(OFFSETS, add_mask=True)
    return torch_em.default_segmentation_loader(paths, "raw", paths, "labels",
                                                args.batch_size, patch_shape,
                                                shuffle=True, num_workers=4*args.batch_size,
                                                label_transform2=label_transform)


# data from
# https://github.com/dlmbl/DL-MBL-2021/blob/main/06_instance_segmentation/datasets/data_epithelia.tar.gz
def train_affinities(args):
    model = get_model()
    patch_shape = [256, 256]
    train_loader = get_loader(args, "train", patch_shape=patch_shape)
    val_loader = get_loader(args, "val", patch_shape=patch_shape)

    loss = torch_em.loss.LossWrapper(
        torch_em.loss.DiceLoss(),
        transform=torch_em.loss.ApplyAndRemoveMask()
    )

    name = "affinity-model"
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
        device=args.device
    )
    trainer.fit(args.n_iterations)


if __name__ == "__main__":
    parser = parser_helper()
    args = parser.parse_args()
    train_affinities(args)
