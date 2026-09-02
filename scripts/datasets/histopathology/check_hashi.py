import os
import argparse

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_hashi_loader


DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "hashi")


def check_hashi(path, save_path):
    # Restrict to a handful of CINJ tiles by default: each cohort's images are a multi-GB
    # archive on Zenodo, and 'sample_ids' picks a shallow subset without touching the rest.
    loader = get_hashi_loader(
        path=path,
        batch_size=1,
        patch_shape=(512, 512),
        cohorts="cinj",
        sample_ids=["10253", "10254", "10255", "10256"],
        download=True,
    )
    check_loader(loader, 4, instance_labels=False, plt=save_path is not None, rgb=True, save_path=save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=DEFAULT_ROOT, help="Folder for the HASHI data.")
    parser.add_argument("--save-path", default=None, help="Optional path for a non-interactive loader check.")
    args = parser.parse_args()
    check_hashi(args.path, args.save_path)


if __name__ == "__main__":
    main()
