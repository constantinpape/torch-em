import os
import sys
import argparse

from torch_em.util.debug import check_loader
from torch_em.data import MinForegroundSampler
from torch_em.data.datasets.electron_microscopy import get_cmito_loader

sys.path.append("..")


def check_cmito(path, stage, patch_shape, n_samples, bounding_box, save_path):
    from util import ROOT

    loader = get_cmito_loader(
        path=os.path.join(ROOT, "cmito") if path is None else path,
        batch_size=1,
        patch_shape=patch_shape,
        stage=stage,
        bounding_box=bounding_box,
        download=True,
        sampler=MinForegroundSampler(min_fraction=0.01),
    )
    check_loader(loader, n_samples, instance_labels=True, plt=save_path is not None, save_path=save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, help="Where to cache the data. Defaults to the 'cmito' folder in ROOT.")
    parser.add_argument("--stage", default="dauer", help="The developmental stage to check.")
    parser.add_argument("--patch_shape", type=int, nargs=3, default=[32, 128, 128], help="The patch shape.")
    parser.add_argument("-n", "--n_samples", type=int, default=8, help="The number of samples to display.")
    parser.add_argument(
        "--bounding_box", type=float, nargs=6, default=None,
        help="A region in nm (x_min x_max y_min y_max z_min z_max) instead of the full stage volume.",
    )
    parser.add_argument("--save_path", default=None, help="Save a matplotlib figure here instead of using napari.")
    args = parser.parse_args()
    check_cmito(args.path, args.stage, tuple(args.patch_shape), args.n_samples, args.bounding_box, args.save_path)


if __name__ == "__main__":
    main()
