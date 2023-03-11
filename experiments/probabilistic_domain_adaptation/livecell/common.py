import argparse

import torch_em
from torch_em.data.datasets import get_livecell_loader
from torchvision import transforms

CELL_TYPES = ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3"]


#
# The augmentations we use for the LiveCELL experiments:
# - weak augmenations: blurring and additive gaussian noise
# - strong augmentations: TODO
#


def weak_augmentations(p=0.25):
    norm = torch_em.transform.raw.standardize
    aug = transforms.Compose([
        norm,
        transforms.RandomApply([torch_em.transform.raw.GaussianBlur()], p=p),
        transforms.RandomApply([torch_em.transform.raw.AdditiveGaussianNoise(
            scale=(0, 0.15), clip_kwargs=False)], p=p
        ),
    ])
    return torch_em.transform.raw.get_raw_transform(normalizer=norm, augmentation1=aug)


#
# Other utility functions: parser, loaders
#


def get_parser(default_batch_size=8, default_iterations=int(1e5)):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-p", "--phase", required=True)
    parser.add_argument("-b", "--batch_size", default=default_batch_size, type=int)
    parser.add_argument("-n", "--n_iterations", default=default_iterations, type=int)
    parser.add_argument("-s", "--save_root")
    parser.add_argument("-c", "--cell_types", nargs="+", default=CELL_TYPES)
    return parser


def get_supervised_loader(args, split, cell_type):
    patch_shape = (512, 512)
    loader = get_livecell_loader(
        args.input, patch_shape, split,
        download=True, binary=True, batch_size=args.batch_size,
        cell_types=[cell_type], num_workers=8, shuffle=True,
    )
    return loader


def get_unsupervised_loader(args, split, cell_type):
    pass
