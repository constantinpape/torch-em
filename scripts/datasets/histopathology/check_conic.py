import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_conic_loader


sys.path.append("..")


def check_conic():
    from util import ROOT

    loader = get_conic_loader(
        path=os.path.join(ROOT, "conic"),
        split="train",
        batch_size=2,
        patch_shape=(1, 512, 512),
        label_choice="instances",
        download=True,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_conic()
