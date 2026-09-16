import os
import sys
import argparse

from torch_em.data.datasets.light_microscopy.pcmmd import get_pcmmd_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_pcmmd(save_path=None):
    from util import ROOT

    path = os.path.join(ROOT, "pcmmd")
    for split in ("train", "test"):
        for label_choice in ("semantic", "binary"):
            loader = get_pcmmd_loader(
                path=path,
                batch_size=2,
                patch_shape=(256, 256),
                split=split,
                cell_type="both",
                label_choice=label_choice,
                download=True,
            )
            if save_path is None:
                split_save_path = None
            else:
                base, extension = os.path.splitext(save_path)
                split_save_path = f"{base}_{split}_{label_choice}{extension or '.png'}"
            check_loader(loader, 2, plt=split_save_path is not None, save_path=split_save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", default=None, help="Save non-interactive previews at this path.")
    args = parser.parse_args()
    check_pcmmd(args.save_path)


if __name__ == "__main__":
    main()
