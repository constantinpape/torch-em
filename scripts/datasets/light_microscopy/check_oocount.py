import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_oocount_loader


sys.path.append("..")


def check_oocount():
    from util import ROOT

    loader = get_oocount_loader(
        path=os.path.join(ROOT, "oocount"),
        batch_size=1,
        patch_shape=(64, 128, 128),
        timepoint="perinatal",
        split="train",
        download=False,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_oocount()
