import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_cellmap_loader


sys.path.append("..")


def check_cellmap():
    # from util import ROOT
    ROOT = "/home/anwai/data"

    loader = get_cellmap_loader(
        path=os.path.join(ROOT, "cellmap-segmentation-challenge"),
        batch_size=2,
        patch_shape=(16, 512, 512),
        ndim=3,
        download=True,
        sampler=MinInstanceSampler(min_num_instances=3),
        crops="234",
        organelles=None,
    )
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_cellmap()
