import os
import sys
import argparse

from torch_em.data.datasets import get_cellular_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_cellular(save_path=None):
    from util import ROOT

    # NOTE: The first call reads the annotated fields out of the archives and packs them into h5
    # files. That preprocessing decodes one full frame image per cell, so it takes a few minutes.
    path = os.path.join(ROOT, "cellular")
    for channel in ("all", "fitc"):
        for label_choice in ("instances", "semantic"):
            loader = get_cellular_loader(
                path=path,
                batch_size=1,
                patch_shape=(512, 512),
                channel=channel,
                label_choice=label_choice,
                download=True,
            )
            if save_path is None:
                split_save_path = None
            else:
                base, extension = os.path.splitext(save_path)
                split_save_path = f"{base}_{channel}_{label_choice}{extension or '.png'}"
            check_loader(
                loader, 2, instance_labels=label_choice == "instances",
                plt=split_save_path is not None, save_path=split_save_path,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", default=None, help="Save non-interactive previews at this path.")
    args = parser.parse_args()
    check_cellular(args.save_path)


if __name__ == "__main__":
    main()
