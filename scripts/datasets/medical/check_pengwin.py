import os
import sys

from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_pengwin_loader


sys.path.append("..")


def check_pengwin():
    # from util import ROOT
    ROOT = "/media/anwai/ANWAI/data"

    loader = get_pengwin_loader(
        path=os.path.join(ROOT, "pengwin"),
        patch_shape=(1, 512, 512),
        ndim=2,
        batch_size=2,
        modality="X-Ray",
        split="train",
        resize_inputs=False,
        download=True,
        sampler=MinInstanceSampler(),
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_pengwin()
