import os
import argparse

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_gleason2019_loader


DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "gleason2019")


def check_gleason2019(path, split, save_path):
    loader = get_gleason2019_loader(
        path=path,
        batch_size=1,
        patch_shape=(512, 512),
        split=split,
        download=True,
        shuffle=True,
    )
    check_loader(loader, 4, instance_labels=False, plt=save_path is not None, rgb=True, save_path=save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=DEFAULT_ROOT, help="Folder for the Gleason2019 data.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="The split to check.")
    parser.add_argument("--save-path", default=None, help="Optional path for a non-interactive loader check.")
    args = parser.parse_args()
    check_gleason2019(args.path, args.split, args.save_path)


if __name__ == "__main__":
    main()
