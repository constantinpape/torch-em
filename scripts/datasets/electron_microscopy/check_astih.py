import os
import sys

from torch_em.data.datasets import get_astih_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_astih():
    from util import ROOT

    loader = get_astih_loader(
        os.path.join(ROOT, "astih"), (512, 512), 1, split="train", download=True, shuffle=True,
    )
    check_loader(loader, 8, plt=True, save_path="./astih.png")


if __name__ == "__main__":
    check_astih()
