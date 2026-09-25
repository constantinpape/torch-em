import os
import sys
import argparse

from torch_em.data.datasets import get_isbi14_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_isbi14(save_path=None):
    from util import ROOT

    path = os.path.join(ROOT, "isbi14")
    configs = [
        ("synthetic", "nucleus", "train"),
        ("synthetic", "cytoplasm", "train"),
        ("real", "nucleus", "train"),
    ]
    for image_source, label_choice, split in configs:
        loader = get_isbi14_loader(
            path=path,
            batch_size=1,
            patch_shape=(256, 256),
            image_source=image_source,
            label_choice=label_choice,
            split=split,
            download=True,
        )
        if save_path is None:
            split_save_path = None
        else:
            base, extension = os.path.splitext(save_path)
            split_save_path = f"{base}_{image_source}_{label_choice}{extension or '.png'}"
        check_loader(loader, 2, instance_labels=True, plt=split_save_path is not None, save_path=split_save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", default=None, help="Save non-interactive previews at this path.")
    args = parser.parse_args()
    check_isbi14(args.save_path)


if __name__ == "__main__":
    main()
