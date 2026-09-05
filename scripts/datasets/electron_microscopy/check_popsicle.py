import os
import sys
import argparse

from torch_em.util.debug import check_loader
from torch_em.data import MinForegroundSampler
from torch_em.data.datasets.electron_microscopy import get_popsicle_loader

sys.path.append("..")


def check_popsicle(split, patch_shape, n_samples):
    from util import ROOT

    loader = get_popsicle_loader(
        path=os.path.join(ROOT, "popsicle"),
        patch_shape=patch_shape,
        batch_size=1,
        split=split,
        download=True,
        sampler=MinForegroundSampler(min_fraction=0.05),
    )
    check_loader(loader, n_samples, instance_labels=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", help="The data split, 'train' or 'test'.")
    parser.add_argument("--patch_shape", type=int, nargs=3, default=[64, 384, 384], help="The patch shape.")
    parser.add_argument("-n", "--n_samples", type=int, default=8, help="The number of samples to display.")
    args = parser.parse_args()
    check_popsicle(args.split, tuple(args.patch_shape), args.n_samples)


if __name__ == "__main__":
    main()
