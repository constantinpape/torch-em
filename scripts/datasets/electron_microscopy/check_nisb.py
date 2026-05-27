import os
import sys

from torch_em.data.datasets import get_nisb_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_nisb():
    from util import ROOT

    loader = get_nisb_loader(
        os.path.join(ROOT, "nisb"), (32, 256, 256), 1, setting="base", split="train", download=True
    )
    check_loader(loader, 8, instance_labels=True)


def main():
    check_nisb()


if __name__ == "__main__":
    main()
