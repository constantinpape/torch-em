import os
import sys
import argparse

from torch_em.util.debug import check_loader
from torch_em.data import MinForegroundSampler
from torch_em.data.datasets.electron_microscopy import get_deepict_pombe_loader
from torch_em.data.datasets.electron_microscopy.deepict_pombe import RUNS, TARGETS

sys.path.append("..")


def check_deepict_pombe(target, runs, patch_shape, n_samples):
    from util import ROOT

    loader = get_deepict_pombe_loader(
        path=os.path.join(ROOT, "deepict_pombe"),
        patch_shape=patch_shape,
        batch_size=1,
        target=target,
        runs=runs,
        download=True,
        sampler=MinForegroundSampler(min_fraction=0.001),
    )
    check_loader(loader, n_samples, instance_labels=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="all", choices=TARGETS, help="The target.")
    parser.add_argument("--runs", nargs="+", default=list(RUNS), help="The runs to use.")
    parser.add_argument("--patch_shape", type=int, nargs=3, default=[32, 256, 256], help="The patch shape.")
    parser.add_argument("-n", "--n_samples", type=int, default=8, help="The number of samples to display.")
    args = parser.parse_args()
    check_deepict_pombe(args.target, args.runs, tuple(args.patch_shape), args.n_samples)


if __name__ == "__main__":
    main()
