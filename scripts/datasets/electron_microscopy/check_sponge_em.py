import os
import sys

from torch_em.data.datasets import get_sponge_em_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_sponge_em():
    from util import ROOT

    loader = get_sponge_em_loader(os.path.join(ROOT, "sponge_em"), "instances", (8, 512, 512), 1, download=True)
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_sponge_em()
