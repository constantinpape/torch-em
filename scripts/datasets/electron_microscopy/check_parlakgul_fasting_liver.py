import os
import argparse

import matplotlib
matplotlib.use("Agg")

from torch_em.data.datasets.electron_microscopy.parlakgul_fasting_liver import (  # noqa
    get_parlakgul_fasting_liver_loader
)
from torch_em.util.debug import check_loader  # noqa

CIDAS_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"

# A central region of the lean fasted sample: 64 z-slices, 1024 x 1024 in xy.
CHECK_BBOX = (3000, 4024, 3000, 4024, 4000, 4064)


def check_parlakgul_fasting_liver(path, sample, bounding_box, label_choice, patch_shape, n_samples, save_path):
    loader = get_parlakgul_fasting_liver_loader(
        path, patch_shape=patch_shape, batch_size=1, bounding_boxes=[bounding_box], sample=sample,
        label_choice=label_choice, download=True,
    )
    check_loader(loader, n_samples, plt=save_path is not None, save_path=save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", default=os.path.join(CIDAS_ROOT, "parlakgul_fasting_liver"), help="Where to cache the data."
    )
    parser.add_argument("--sample", default="9428", help="The liver sample: 9428, 8970, 2679 or 1858.")
    parser.add_argument(
        "--bounding_box", type=int, nargs=6, default=CHECK_BBOX,
        help="The region in voxels (x_min x_max y_min y_max z_min z_max).",
    )
    parser.add_argument(
        "--label_choice", default="mito", help="The labels: er, mito, lipid_droplet, nuclear_membrane or cells."
    )
    parser.add_argument("--patch_shape", type=int, nargs=3, default=[32, 512, 512], help="The patch shape.")
    parser.add_argument("-n", "--n_samples", type=int, default=8, help="The number of samples to display.")
    parser.add_argument("--save_path", default=None, help="Save a matplotlib figure here instead of using napari.")
    args = parser.parse_args()
    check_parlakgul_fasting_liver(
        args.path, args.sample, tuple(args.bounding_box), args.label_choice, tuple(args.patch_shape),
        args.n_samples, args.save_path,
    )


if __name__ == "__main__":
    main()
