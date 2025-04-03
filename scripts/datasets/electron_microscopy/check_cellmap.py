import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_cellmap_loader


sys.path.append("..")


def check_cellmap():
    # from util import ROOT
    ROOT = "/media/anwai/ANWAI/data"

    loader = get_cellmap_loader(
        path=os.path.join(ROOT, "cellmap-segmentation-challenge"),
        batch_size=2,
        patch_shape=(8, 256, 256),
        download=True,
        sampler=MinInstanceSampler(),
        crops=["234", "235"],
    )
    check_loader(loader, 8)


if __name__ == "__main__":
    check_cellmap()
