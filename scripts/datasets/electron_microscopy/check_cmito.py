import os
import sys
import argparse

from torch_em.util.debug import check_loader
from torch_em.data import MinForegroundSampler
from torch_em.data.datasets.electron_microscopy import get_cmito_loader

sys.path.append("..")


def check_cmito(stage, patch_shape, n_samples):
    from util import ROOT

    loader = get_cmito_loader(
        path=os.path.join(ROOT, "cmito"),
        batch_size=1,
        patch_shape=patch_shape,
        stage=stage,
        download=True,
        sampler=MinForegroundSampler(min_fraction=0.01),
    )
    check_loader(loader, n_samples, instance_labels=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="dauer", help="The developmental stage to check.")
    parser.add_argument("--patch_shape", type=int, nargs=3, default=[32, 128, 128], help="The patch shape.")
    parser.add_argument("-n", "--n_samples", type=int, default=8, help="The number of samples to display.")
    args = parser.parse_args()
    check_cmito(args.stage, tuple(args.patch_shape), args.n_samples)


if __name__ == "__main__":
    main()
