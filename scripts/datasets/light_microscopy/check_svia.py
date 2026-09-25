import os
import sys
import argparse

from torch_em.data.datasets import get_svia_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_svia(save_path=None):
    from util import ROOT

    # NOTE: The archive is a rar file, so the extraction needs 'rarfile' and the 'unrar' program.
    path = os.path.join(ROOT, "svia")
    for split in ("train", "val", "test"):
        loader = get_svia_loader(
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
            loader, 2, instance_labels=True, plt=split_save_path is not None, save_path=split_save_path
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", default=None, help="Save non-interactive split previews at this path.")
    args = parser.parse_args()
    check_svia(args.save_path)


if __name__ == "__main__":
    main()
