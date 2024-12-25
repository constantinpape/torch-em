import os
import sys

from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_mice_tumseg_loader


sys.path.append("..")


def check_mice_tumseg():
    # from util import ROOT
    ROOT = "/media/anwai/ANWAI/data"

    loader = get_mice_tumseg_loader(
        path=os.path.join(ROOT, "mice_tumseg"),
        batch_size=1,
        patch_shape=(1, 512, 512),
        ndim=2,
        split="train",
        resize_inputs=True,
        download=True,
        sampler=MinInstanceSampler(),
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_mice_tumseg()
