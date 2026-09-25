import os
import sys
import argparse

from torch_em.data.datasets import get_nucverse3d_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_nucverse3d(save_path=None):
    from util import ROOT

    path = os.path.join(ROOT, "nucverse3d")
    for dataset in ("liver", "liver_hcc", "drosophila_glia"):
        for split in ("train", "test"):
            loader = get_nucverse3d_loader(
                path=path,
                batch_size=1,
                patch_shape=(32, 256, 256),
                dataset=dataset,
                split=split,
                download=True,
            )
            if save_path is None:
                split_save_path = None
            else:
                base, extension = os.path.splitext(save_path)
                split_save_path = f"{base}_{dataset}_{split}{extension or '.png'}"
            check_loader(
                loader, 2, instance_labels=True, plt=split_save_path is not None, save_path=split_save_path
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", default=None, help="Save non-interactive previews at this path.")
    args = parser.parse_args()
    check_nucverse3d(args.save_path)


if __name__ == "__main__":
    main()
