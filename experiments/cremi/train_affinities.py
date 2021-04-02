import argparse
import os
import numpy as np

import torch_em
from torch_em.model import AnisotropicUNet
from torch_em.loss import DiceLoss, LossWrapper, ApplyAndRemoveMask

ROOT = '/scratch/pape/cremi'
OFFSETS = [
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
    [-2, 0, 0],
    [0, -3, 0],
    [0, 0, -3],
    [-3, 0, 0],
    [0, -9, 0],
    [0, 0, -9],
    [-4, 0, 0],
    [0, -27, 0],
    [0, 0, -27]
]


def samples_to_paths(samples, is_train):
    val_slice = 75
    if is_train:
        # we train on sampleA + sampleB + 0:75 of sampleC
        paths = [os.path.join(ROOT, f'sample_{sample}_20160501.hdf') for sample in samples]
        if len(samples) == 3:
            rois = [np.s_[:, :, :], np.s_[:, :, :], np.s_[:val_slice, :, :]]
        elif 'C' in samples:
            rois = [np.s_[:, :, :], np.s_[:val_slice, :, :]]
        else:
            rois = [np.s_[:, :, :], np.s_[:, :, :]]
    else:
        # we validate on 75:125 of sampleC, regardless of inut samples
        paths = [
            os.path.join(ROOT, 'sample_C_20160501.hdf')
        ]
        rois = [np.s_[val_slice:, :, :]]
    return paths, rois


def get_loader(samples, is_train, patch_shape,
               batch_size=1, n_samples=None,
               roi=None):

    paths, rois = samples_to_paths(samples, is_train)

    raw_key = 'volumes/raw'
    label_key = 'volumes/labels/neuron_ids'

    p_drop_slice = 0.025
    raw_transform = torch_em.transform.get_raw_transform(
        augmentation1=torch_em.transform.EMDefectAugmentation(p_drop_slice=p_drop_slice)
    )
    label_transform = torch_em.transform.label.AffinityTransform(offsets=OFFSETS, add_mask=True)

    sampler = None
    # sampler = torch_em.data.MinForegroundSampler(min_fraction=0.05, p_reject=.75)

    return torch_em.default_segmentation_loader(
        paths, raw_key,
        paths, label_key,
        batch_size=batch_size,
        patch_shape=patch_shape,
        rois=rois,
        raw_transform=raw_transform,
        label_transform2=label_transform,
        sampler=sampler,
        n_samples=n_samples,
        num_workers=8*batch_size,
        shuffle=True
    )


def get_model(large_model):
    n_out = len(OFFSETS)
    if large_model:
        print("Using large model")
        model = AnisotropicUNet(
            scale_factors=[
                [1, 3, 3],
                [1, 3, 3],
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
                [1, 3, 3],
                [1, 3, 3],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2]
            ],
            in_channels=1,
            out_channels=n_out,
            initial_features=32,
            gain=2,
            final_activation='Sigmoid'
        )
    return model


def normalize_samples(samples):
    if set(samples) == {'A', 'B', 'C'}:
        samples = ('A', 'B', 'C')
        prefix = None
    elif set(samples) == {'A', 'B'}:
        samples = ('A', 'B')
        prefix = 'AB'
    elif set(samples) == {'A', 'C'}:
        samples = ('A', 'C')
        prefix = 'AC'
    elif set(samples) == {'B', 'C'}:
        samples = ('B', 'C')
        prefix = 'BC'
    return samples, prefix


def train_affinities(args):
    large_model = bool(args.large_model)
    model = get_model(large_model)

    # patch shapes:
    if large_model:
        # TODO determine patch shape
        # largest possible shape for A100 with mixed training and large model
        # patch_shape = [32, 320, 320]
        patch_shape = [32, 256, 256]
    else:
        patch_shape = [32, 360, 360]

    samples, prefix = normalize_samples(args.samples)
    train_loader = get_loader(
        samples=samples,
        is_train=True,
        patch_shape=patch_shape,
        n_samples=1000
    )
    val_loader = get_loader(
        samples=samples,
        is_train=False,
        patch_shape=patch_shape,
        n_samples=100
    )

    loss = LossWrapper(loss=DiceLoss(),
                       transform=ApplyAndRemoveMask())

    tag = 'large' if large_model else 'default'
    if prefix is not None:
        tag += f"_{prefix}"
    name = f"affinity_model_{tag}"
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
        trainer.fit(args.iterations, 'latest')
    else:
        trainer.fit(args.iterations)


def check(train=True, val=True, n_images=2):
    from torch_em.util.debug import check_loader
    patch_shape = [32, 256, 256]
    if train:
        print("Check train loader")
        loader = get_loader(True, patch_shape)
        check_loader(loader, n_images)
    if val:
        print("Check val loader")
        loader = get_loader(False, patch_shape)
        check_loader(loader, n_images)


# TODO advanced cremi training (probably should make 'train_advanced.py' for this)
# - more defect and mis-alignment augmentations
# -- pasting defect patches
# -- simulating contrast defect
# -- simulating tear defect
# -- alignment jitter
# - more augmentations
# -- elastic
# - training with aligned volumes (needs ignore mask)
# - training with glia mask
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', '-c', type=int, default=0)
    parser.add_argument('--iterations', '-i', type=int, default=int(1e5))
    parser.add_argument('--from_checkpoint', type=int, default=0)
    parser.add_argument('--large_model', '-l', type=int, default=0)
    parser.add_argument('--samples', type=str, nargs='+', default=['A', 'B', 'C'])

    args = parser.parse_args()

    if args.check:
        check(train=True, val=True)
    else:
        train_affinities(args)
