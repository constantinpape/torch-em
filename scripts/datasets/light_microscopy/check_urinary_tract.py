import os
import sys
import argparse

from torch_em.data.datasets import get_urinary_tract_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_urinary_tract(save_path=None):
    from util import ROOT

    path = os.path.join(ROOT, "urinary_tract")
    for split in ("train", "validation", "test"):
        for label_choice in ("semantic", "binary"):
            loader = get_urinary_tract_loader(
                path=path,
                batch_size=1,
                patch_shape=(512, 512),
                split=split,
                label_choice=label_choice,
                download=True,
            )
            if save_path is None:
                split_save_path = None
            else:
                base, extension = os.path.splitext(save_path)
                split_save_path = f"{base}_{split}_{label_choice}{extension or '.png'}"
            check_loader(
                loader, 2, instance_labels=False, plt=split_save_path is not None, save_path=split_save_path
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", default=None, help="Save non-interactive split previews at this path.")
    args = parser.parse_args()
    check_urinary_tract(args.save_path)


if __name__ == "__main__":
    main()
