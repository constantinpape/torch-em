import os
import sys
import argparse

from torch_em.data.datasets import get_nuc_morph_timelapse_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_nuc_morph_timelapse(save_path=None):
    from util import ROOT

    path = os.path.join(ROOT, "nuc_morph_timelapse")
    loader = get_nuc_morph_timelapse_loader(
        path=path,
        batch_size=1,
        patch_shape=(32, 512, 512),
        colony="20200323_09_small",
        timepoints=[0, 285],
        download=True,
    )
    check_loader(loader, 3, instance_labels=True, plt=save_path is not None, save_path=save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", default=None, help="Save a non-interactive preview at this path.")
    args = parser.parse_args()
    check_nuc_morph_timelapse(args.save_path)


if __name__ == "__main__":
    main()
