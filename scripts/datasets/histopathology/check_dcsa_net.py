import os
import argparse

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_dcsa_net_loader


DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "dcsa_net")


def check_dcsa_net(path, save_path):
    loader = get_dcsa_net_loader(
        path=path,
        batch_size=1,
        patch_shape=(512, 512),
        split="train",
        download=True,
        shuffle=True,
    )
    check_loader(loader, 4, instance_labels=False, plt=save_path is not None, rgb=True, save_path=save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=DEFAULT_ROOT, help="Folder for the DCSA-Net data.")
    parser.add_argument("--save-path", default=None, help="Optional path for a non-interactive loader check.")
    args = parser.parse_args()
    check_dcsa_net(args.path, args.save_path)


if __name__ == "__main__":
    main()
