import os
import sys
import argparse

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.electron_microscopy import get_saber_loader

sys.path.append("..")


def check_saber(scale, patch_shape, n_samples):
    from util import ROOT

    loader = get_saber_loader(
        path=os.path.join(ROOT, "saber"),
        patch_shape=patch_shape,
        batch_size=1,
        scale=scale,
        download=True,
        sampler=MinInstanceSampler(),
    )
    check_loader(loader, n_samples, instance_labels=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=0, help="The resolution level of the multiscale data.")
    parser.add_argument("--patch_shape", type=int, nargs=3, default=[64, 384, 384], help="The patch shape.")
    parser.add_argument("-n", "--n_samples", type=int, default=8, help="The number of samples to display.")
    args = parser.parse_args()
    check_saber(args.scale, tuple(args.patch_shape), args.n_samples)


if __name__ == "__main__":
    main()
