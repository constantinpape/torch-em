import os
import sys
import argparse

from torch_em.data.datasets import get_cellapp_loader
from torch_em.util.debug import check_loader

sys.path.append("..")

SUBSETS = [
    ("general", None), ("hela", "test"), ("rpe1", "train"),
    ("rpe1", "test"), ("u2os", "train"), ("u2os", "test"),
]


def check_cellapp(save_path=None):
    from util import ROOT

    path = os.path.join(ROOT, "cellapp")
    for source, split in SUBSETS:
        loader = get_cellapp_loader(
            path=path,
            batch_size=1,
            patch_shape=(512, 512),
            source=source,
            split=split,
            size=1,
            download=True,
        )
        if save_path is None:
            split_save_path = None
        else:
            base, extension = os.path.splitext(save_path)
            split_save_path = f"{base}_{source}_{split}{extension or '.png'}"
        check_loader(
            loader, 3, instance_labels=True, plt=split_save_path is not None, save_path=split_save_path
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", default=None, help="Save non-interactive subset previews at this path.")
    args = parser.parse_args()
    check_cellapp(args.save_path)


if __name__ == "__main__":
    main()
