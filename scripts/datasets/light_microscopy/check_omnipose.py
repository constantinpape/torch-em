import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_omnipose_loader

sys.path.append("..")


def check_omnipose():
    # from util import ROOT
    ROOT = "/media/anwai/ANWAI/data"

    loader = get_omnipose_loader(
        path=os.path.join(ROOT, "omnipose"),
        batch_size=1,
        patch_shape=(1024, 1024),
        split="train",
        data_choice=None,
        sampler=MinInstanceSampler(),
        shuffle=True,
        download=True,
    )
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_omnipose()
