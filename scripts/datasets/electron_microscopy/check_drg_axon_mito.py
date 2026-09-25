import os
import sys
import argparse

from torch_em.util.debug import check_loader
from torch_em.data import MinForegroundSampler
from torch_em.data.datasets.electron_microscopy import get_drg_axon_mito_loader

sys.path.append("..")


def check_drg_axon_mito(patch_shape, n_samples):
    from util import ROOT

    loader = get_drg_axon_mito_loader(
        path=os.path.join(ROOT, "drg_axon_mito"),
        patch_shape=patch_shape,
        batch_size=1,
        download=True,
        sampler=MinForegroundSampler(min_fraction=0.05),
    )
    check_loader(loader, n_samples, instance_labels=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_shape", type=int, nargs=3, default=[64, 384, 384], help="The patch shape.")
    parser.add_argument("-n", "--n_samples", type=int, default=8, help="The number of samples to display.")
    args = parser.parse_args()
    check_drg_axon_mito(tuple(args.patch_shape), args.n_samples)


if __name__ == "__main__":
    main()
