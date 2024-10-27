import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_vicar_loader

sys.path.append("..")


def check_vicar():
    # from util import ROOT
    ROOT = "/media/anwai/ANWAI/data"

    loader = get_vicar_loader(
        path=os.path.join(ROOT, "vicar"),
        patch_shape=(512, 512),
        batch_size=1,
        download=True,
    )
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_vicar()
