import argparse
import os
from glob import glob
from functools import partial

import torch
import torch_em
from elf.io import open_file
from torch_em.model import UNet2d

ROOT_TRAIN = '/g/kreshuk/wolny/Datasets/Ovules/GT2x/train'
ROOT_VAL = '/g/kreshuk/wolny/Datasets/Ovules/GT2x/val'


# exclude the volumes that don't fit
def get_paths(split, patch_shape, raw_key):
    root = ROOT_TRAIN if split == 'train' else ROOT_VAL
    paths = glob(os.path.join(root, '*.h5'))

    paths = [p for p in paths if all(
        sh >= psh for sh, psh in zip(open_file(p, 'r')[raw_key].shape, patch_shape)
    )]
    return paths


def get_loader(split, patch_shape, batch_size,
               n_samples=None, roi=None):
    raw_key = 'raw'
    label_key = 'label'
    paths = get_paths(split, patch_shape, raw_key)

    sampler = torch_em.data.MinForegroundSampler(min_fraction=0.3, p_reject=1.)
    label_transform = partial(torch_em.transform.label.connected_components, ensure_zero=True)

    return torch_em.default_segmentation_loader(
        paths, raw_key,
        paths, label_key,
        batch_size=batch_size,
        patch_shape=patch_shape,
        label_transform=label_transform,
        sampler=sampler,
        n_samples=n_samples,
        num_workers=8*batch_size,
        shuffle=True,
        label_dtype=torch.int64,
        ndim=2
    )


def get_model():
    n_out = 12
    model = UNet2d(
        in_channels=1,
        out_channels=n_out,
        initial_features=64,
        gain=2,
        depth=4,
        final_activation=None
    )
    return model


def train_contrastive(args):
    model = get_model()
    patch_shape = [1, 384, 384]

    train_loader = get_loader(
        split='train',
        patch_shape=patch_shape,
        batch_size=1,
        n_samples=2500
    )
    val_loader = get_loader(
        split='val',
        patch_shape=patch_shape,
        batch_size=1,
        n_samples=100
    )

    loss = torch_em.loss.ContrastiveLoss(
        delta_var=.75,
        delta_dist=2.,
        impl=args.impl
    )

    name = "embedding_model2d_" + args.impl
    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=loss,
        learning_rate=5e-5,
        mixed_precision=True,
        log_image_interval=50
    )

    if args.from_checkpoint:
        trainer.fit(args.iterations, 'latest')
    else:
        trainer.fit(args.iterations)


def check(train=True, val=True, n_images=5):
    from torch_em.util.debug import check_loader
    patch_shape = [1, 512, 512]
    if train:
        print("Check train loader")
        loader = get_loader('train', patch_shape, batch_size=1)
        check_loader(loader, n_images)
    if val:
        print("Check val loader")
        loader = get_loader('val', patch_shape, batch_size=1)
        check_loader(loader, n_images)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--impl', '-i', default='scatter')
    parser.add_argument('--check', '-c', type=int, default=0)
    parser.add_argument('--iterations', '-n', type=int, default=int(1e5))
    parser.add_argument('--from_checkpoint', type=int, default=0)

    args = parser.parse_args()
    if args.check:
        check(train=True, val=True)
    else:
        train_contrastive(args)
