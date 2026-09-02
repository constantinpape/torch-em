import os
import argparse

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_crc_epithelium_loader


DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "crc_epithelium")


def check_crc_epithelium(path, save_path):
    # Restrict to a handful of H&E cores by default: each stain is a multi-GB archive on
    # DataverseNO, and 'sample_ids' picks a shallow subset without touching the rest.
    loader = get_crc_epithelium_loader(
        path=path,
        batch_size=1,
        patch_shape=(512, 512),
        stains="HE",
        sample_ids=["A001-4", "A002-5", "A099-3", "A100-1"],
        download=True,
    )
    check_loader(loader, 4, instance_labels=False, plt=save_path is not None, rgb=True, save_path=save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=DEFAULT_ROOT, help="Folder for the CRC epithelium data.")
    parser.add_argument("--save-path", default=None, help="Optional path for a non-interactive loader check.")
    args = parser.parse_args()
    check_crc_epithelium(args.path, args.save_path)


if __name__ == "__main__":
    main()
