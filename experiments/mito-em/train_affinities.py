import argparse
import os

import torch_em
from torch_em.model import AnisotropicUNet
from torch_em.loss import DiceLoss, LossWrapper, ApplyAndRemoveMask

ROOT = '/scratch/pape/mito_em/data'
OFFSETS = [
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
    [-2, 0, 0],
    [0, -3, 0],
    [0, 0, -3],
    [-3, 0, 0],
    [0, -9, 0],
    [0, 0, -9]
]


def get_loader(datasets, patch_shape,
               batch_size=1, n_samples=None,
               roi=None):

    paths = [
        os.path.join(ROOT, f'{ds}.n5') for ds in datasets
    ]

    raw_key = 'raw'
    label_key = 'labels'

    sampler = torch_em.data.MinForegroundSampler(min_fraction=0.05, p_reject=.75)
    label_transform = torch_em.transform.label.AffinityTransform(offsets=OFFSETS,
                                                                 add_binary_target=True,
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
        shuffle=True
    )


def get_model(large_model):
    n_out = len(OFFSETS) + 1
    if large_model:
        print("Using large model")
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
            initial_features=128,
            gain=2,
            final_activation='Sigmoid'
        )
    else:
        print("Using vanilla model")
        model = AnisotropicUNet(
            scale_factors=[
                [1, 2, 2],
                [1, 2, 2],
                [2, 2, 2],
                [2, 2, 2]
            ],
            in_channels=1,
            out_channels=n_out,
            initial_features=64,
            gain=2,
            final_activation='Sigmoid'
        )
    return model


# TODO bce + dice; need dice with sigmoid?!
def train_affinities(args, datasets):
    large_model = bool(args.large_model)
    model = get_model(large_model)

    # patch shapes:
    if large_model:
        # largest possible shape for A100 with mixed training and large model
        patch_shape = [32, 256, 256]
    else:
        # largest possible shape for 2080Ti with mixed training
        # patch_shape = [32, 320, 320]
        patch_shape = [32, 256, 256]

    train_sets = [f'{ds}_train' for ds in datasets]
    val_sets = [f'{ds}_val' for ds in datasets]
    if args.train_on_val:
        train_sets += val_sets

    train_loader = get_loader(
        datasets=train_sets,
        patch_shape=patch_shape,
        n_samples=1000
    )
    val_loader = get_loader(
        datasets=val_sets,
        patch_shape=patch_shape,
        n_samples=100
    )

    loss = LossWrapper(loss=DiceLoss(),
                       transform=ApplyAndRemoveMask())

    tag = 'large' if large_model else 'default'
    if args.train_on_val:
        tag += '_train_on_val'
    name = f"affinity_model_{tag}_{'_'.join(datasets)}"
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


def check(datasets, train=True, val=True, n_images=5):
    from torch_em.util.debug import check_loader
    patch_shape = [32, 256, 256]
    if train:
        print("Check train loader")
        dsets = [f'{ds}_train' for ds in datasets]
        loader = get_loader(dsets, patch_shape)
        check_loader(loader, n_images)
    if val:
        print("Check val loader")
        dsets = [f'{ds}_val' for ds in datasets]
        loader = get_loader(dsets, patch_shape)
        check_loader(loader, n_images)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', '-d', type=str, nargs='+', default=['human', 'rat'])
    parser.add_argument('--check', '-c', type=int, default=0)
    parser.add_argument('--iterations', '-i', type=int, default=int(1e5))
    parser.add_argument('--large_model', '-l', type=int, default=0)
    parser.add_argument('--from_checkpoint', type=int, default=0)
    parser.add_argument('--train_on_val', type=int, default=0)

    dataset_names = ['human', 'rat']

    args = parser.parse_args()
    datasets = args.datasets
    datasets.sort()
    assert all(ds in dataset_names for ds in datasets)

    if args.check:
        check(datasets, train=True, val=True)
    else:
        train_affinities(args, datasets)
