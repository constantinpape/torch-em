import os
import sys

from torch_em.data.datasets import get_isbi_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_isbi():
    from util import ROOT

    data_path = os.path.join(ROOT, "isbi.h5")
    loader = get_isbi_loader(data_path, (8, 512, 512), 1, download=True)
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_isbi()
