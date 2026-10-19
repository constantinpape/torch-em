import os
import sys

from torch_em.data.datasets import get_perez_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_perez():
    from util import ROOT

    loader = get_perez_loader(
        path=os.path.join(ROOT, "perez"),
        batch_size=1,
        patch_shape=(512, 512),
        split="train",
        organelle="mito",
        download=True,
    )
    check_loader(loader, 8)


if __name__ == "__main__":
    check_perez()
