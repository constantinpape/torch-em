import os
import sys
import argparse

from torch_em.data.datasets import get_fluo_neuronal_cells_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_fluo_neuronal_cells(save_path=None):
    from util import ROOT

    path = os.path.join(ROOT, "fluo_neuronal_cells")
    for collection in ("green", "yellow", "red"):
        for split in ("trainval", "test"):
            loader = get_fluo_neuronal_cells_loader(
                path=path,
                batch_size=1,
                patch_shape=(512, 512),
                split=split,
                collection=collection,
                download=True,
            )
            if save_path is None:
                split_save_path = None
            else:
                base, extension = os.path.splitext(save_path)
                split_save_path = f"{base}_{collection}_{split}{extension or '.png'}"
            check_loader(
                loader, 2, instance_labels=True, plt=split_save_path is not None, save_path=split_save_path
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", default=None, help="Save non-interactive previews at this path.")
    args = parser.parse_args()
    check_fluo_neuronal_cells(args.save_path)


if __name__ == "__main__":
    main()
