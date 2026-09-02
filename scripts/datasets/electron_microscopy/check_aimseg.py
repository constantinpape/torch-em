import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_aimseg_loader


sys.path.append("..")


def check_aimseg():
    from util import ROOT

    loader = get_aimseg_loader(
        path=os.path.join(ROOT, "aimseg"),
        batch_size=2,
        patch_shape=(512, 512),
        targets="semantic",
        download=True,
    )
    check_loader(loader, 8)

    breakpoint()


if __name__ == "__main__":
    check_aimseg()
