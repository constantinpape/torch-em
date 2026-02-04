import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_oocyteseg_loader


sys.path.append("..")


def check_oocyteseg():
    from util import ROOT

    loader = get_oocyteseg_loader(
        path=os.path.join(ROOT, "oocyte"),
        batch_size=1,
        patch_shape=(256, 256),
        split="train",
        species="mouse",
        download=True,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_oocyteseg()
