import os
import sys

from torch_em.data.datasets import (
    get_neurips_cellseg_supervised_loader, get_neurips_cellseg_unsupervised_loader
)
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_neurips():
    from util import ROOT

    neurips_root = os.path.join(ROOT, "neurips-cell-seg")

    loader = get_neurips_cellseg_supervised_loader(neurips_root, "train", (512, 512), 1, download=True)
    check_loader(loader, 15, instance_labels=True, rgb=True)

    loader = get_neurips_cellseg_unsupervised_loader(neurips_root, (512, 512), 1, download=True)
    check_loader(loader, 15, rgb=True)


if __name__ == "__main__":
    check_neurips()
