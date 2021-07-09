import os
from glob import glob

import torch
import torch_em
from elf.io import open_file
from torch_em.model import UNet2d
from torch_em.util import parser_helper

ROOT_TRAIN = '/g/kreshuk/wolny/Datasets/Ovules/GT2x/train'
ROOT_VAL = '/g/kreshuk/wolny/Datasets/Ovules/GT2x/val'
OFFSETS = [
    [-1, 0], [0, -1],
    [-3, 0], [0, -3],
    [-9, 0], [0, -9],
    [-27, 0], [0, -27]
]


# exclude the volumes that don't fit
def get_paths(split, patch_shape, raw_key):
    root = ROOT_TRAIN if split == 'train' else ROOT_VAL
    paths = glob(os.path.join(root, '*.h5'))

    paths = [p for p in paths if all(
        sh >= psh for sh, psh in zip(open_file(p, 'r')[raw_key].shape, patch_shape)
    )]
    return paths


# TODO implement and use plantseg datasets
def get_loader(split, patch_shape, batch_size,
               n_samples=None, roi=None):
    raw_key = 'raw'
    label_key = 'label'
    paths = get_paths(split, patch_shape, raw_key)

    sampler = torch_em.data.MinForegroundSampler(min_fraction=0.1, p_reject=1.)
    label_transform = torch_em.transform.label.AffinityTransform(offsets=OFFSETS,
                                                                 ignore_label=None,
                                                                 add_binary_target=False,
                                                                 add_mask=True)
    return torch_em.default_segmentation_loader(
        paths, raw_key,
        paths, label_key,
        batch_size=batch_size,
        patch_shape=patch_shape,
        label_transform2=label_transform,
        sampler=sampler,
        n_samples=n_samples,
        num_workers=8*batch_size,
        shuffle=True,
        label_dtype=torch.float32,
        ndim=2
    )


def get_model(n_out):
    model = UNet2d(
        in_channels=1,
        out_channels=n_out,
        initial_features=64,
        gain=2,
        depth=4,
        final_activation=None
    )
    return model


# TODO don't hard-code input but take values from command line
def train_affinties(args):
    model = get_model(len(OFFSETS))
    patch_shape = [1, 736, 688]
    batch_size = 4

    train_loader = get_loader(
        split='train',
        patch_shape=patch_shape,
        batch_size=batch_size,
        n_samples=2500
    )
    val_loader = get_loader(
        split='val',
        patch_shape=patch_shape,
        batch_size=1,
        n_samples=100
    )

    loss = torch_em.loss.LossWrapper(
        torch_em.loss.DiceLoss(),
        transform=torch_em.loss.ApplyAndRemoveMask()
    )

    name = "affinity_model2d"
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
        trainer.fit(args.n_iterations, 'latest')
    else:
        trainer.fit(args.n_iterations)


if __name__ == '__main__':
    parser = parser_helper()
    args = parser.parse_args()
    train_affinties(args)
