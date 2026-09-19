import os
import sys
import argparse

from torch_em.util.debug import check_loader
from torch_em.data import MinForegroundSampler
from torch_em.data.datasets.electron_microscopy import get_openorganelle_lipid_droplet_loader
from torch_em.data.datasets.electron_microscopy.openorganelle_lipid_droplet import DATASETS

sys.path.append("..")


def check_openorganelle_lipid_droplet(dataset_names, patch_shape, n_samples):
    from util import ROOT

    loader = get_openorganelle_lipid_droplet_loader(
        path=os.path.join(ROOT, "openorganelle_lipid_droplet"),
        patch_shape=patch_shape,
        batch_size=1,
        dataset_names=dataset_names,
        download=True,
        sampler=MinForegroundSampler(min_fraction=0.01),
    )
    check_loader(loader, n_samples, instance_labels=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_names", nargs="+", default=None, help=f"The datasets to check. One or more of {list(DATASETS)}."
    )
    parser.add_argument("--patch_shape", type=int, nargs=3, default=[32, 128, 128], help="The patch shape.")
    parser.add_argument("-n", "--n_samples", type=int, default=8, help="The number of samples to display.")
    args = parser.parse_args()
    check_openorganelle_lipid_droplet(args.dataset_names, tuple(args.patch_shape), args.n_samples)


if __name__ == "__main__":
    main()
