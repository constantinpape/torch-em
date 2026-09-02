import os
import sys

from torch_em.data.datasets import get_cellseg_3d_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_cellseg_3d():
    from util import ROOT

    loader = get_cellseg_3d_loader(os.path.join(ROOT, "cellseg_3d"), (32, 256, 256), batch_size=1, download=True)
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_cellseg_3d()
