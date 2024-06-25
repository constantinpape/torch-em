import os
import sys

from torch_em.data.datasets import get_dsb_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_dsb():
    from util import ROOT

    loader = get_dsb_loader(os.path.join(ROOT, "dsb"), "train", (256, 256), 1, download=True)
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_dsb()
