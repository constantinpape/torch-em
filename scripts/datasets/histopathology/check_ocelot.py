import os
import argparse

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_ocelot_loader


DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ocelot")


def check_ocelot(path, save_path):
    loader = get_ocelot_loader(
        path=path,
        patch_shape=(512, 512),
        batch_size=1,
        download=True,
        shuffle=True,
    )
    check_loader(loader, 4, instance_labels=False, plt=save_path is not None, rgb=True, save_path=save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=DEFAULT_ROOT, help="Folder for the OCELOT data.")
    parser.add_argument("--save-path", default=None, help="Optional path for a non-interactive loader check.")
    args = parser.parse_args()
    check_ocelot(args.path, args.save_path)


if __name__ == "__main__":
    main()
