import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_bitdepth_nucseg_loader

sys.path.append("..")


def check_bitdepth_nucseg():
    # from util import ROOT
    ROOT = "/media/anwai/ANWAI/data"

    loader = get_bitdepth_nucseg_loader(
        path=os.path.join(ROOT, "bitdepth_nucseg"),
        patch_shape=(512, 512),
        batch_size=2,
        magnification=None,
        download=True,
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_bitdepth_nucseg()
