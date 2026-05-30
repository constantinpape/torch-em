import os

import matplotlib
matplotlib.use("Agg")

from torch_em.data.datasets import get_nisb_loader  # noqa
from torch_em.util.debug import check_loader  # noqa

CIDAS_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_nisb():
    loader = get_nisb_loader(
        os.path.join(CIDAS_ROOT, "nisb"), (32, 256, 256), 1, setting="base", split="train", download=True
    )
    check_loader(loader, 8, instance_labels=True, plt=True, save_path="./check_nisb.png")


def main():
    check_nisb()


if __name__ == "__main__":
    main()
