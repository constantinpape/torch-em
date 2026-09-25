import os
import sys
import argparse

from torch_em.data.datasets import get_apacs23_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_apacs23(save_path=None):
    from util import ROOT

    # NOTE: The first call downloads about 7000 files one by one, which takes a while.
    path = os.path.join(ROOT, "apacs23")
    for split in ("train", "test"):
        loader = get_apacs23_loader(
            path=path,
            batch_size=1,
            patch_shape=(512, 512),
            split=split,
            download=True,
        )
        if save_path is None:
            split_save_path = None
        else:
            base, extension = os.path.splitext(save_path)
            split_save_path = f"{base}_{split}{extension or '.png'}"
        check_loader(
            loader, 2, instance_labels=False, plt=split_save_path is not None, save_path=split_save_path
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", default=None, help="Save non-interactive split previews at this path.")
    args = parser.parse_args()
    check_apacs23(args.save_path)


if __name__ == "__main__":
    main()
