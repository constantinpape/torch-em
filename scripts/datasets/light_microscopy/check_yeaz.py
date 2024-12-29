import os
import sys

from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_yeaz_loader


sys.path.append("..")


def check_yeaz():
    # from util import ROOT
    ROOT = "/media/anwai/ANWAI/data"

    loader = get_yeaz_loader(
        path=os.path.join(ROOT, "yeaz"),
        batch_size=2,
        patch_shape=(512, 512),
        ndim=2,
        choice="phc",  # choose from 'bf' / 'phc'
        split="val",
        sampler=MinInstanceSampler(),
        download=False,
        shuffle=True,
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_yeaz()
