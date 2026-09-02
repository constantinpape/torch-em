import os
import argparse

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_beetle_loader


DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "beetle")


def check_beetle(path, split, sample_ids, save_path):
    loader = get_beetle_loader(
        path=path,
        batch_size=1,
        patch_shape=(512, 512),
        split=split,
        sample_ids=sample_ids,
        resolution_level=2,
        download=True,
        shuffle=True,
    )
    check_loader(loader, 4, instance_labels=False, plt=save_path is not None, rgb=True, save_path=save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=DEFAULT_ROOT, help="Folder for the BEETLE data.")
    parser.add_argument("--split", default="val", choices=["train", "val"], help="The split to check.")
    parser.add_argument("--sample-ids", nargs="+", default=None, help="Optional slide names to restrict to.")
    parser.add_argument("--save-path", default=None, help="Optional path for a non-interactive loader check.")
    args = parser.parse_args()
    check_beetle(args.path, args.split, args.sample_ids, args.save_path)


if __name__ == "__main__":
    main()
