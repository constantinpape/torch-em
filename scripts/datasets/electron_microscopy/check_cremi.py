import os
import sys

from torch_em.data.datasets import get_cremi_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_cremi():
    from util import ROOT

    loader = get_cremi_loader(os.path.join(ROOT, "cremi"), (8, 512, 512), 1, download=True)
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_cremi()
