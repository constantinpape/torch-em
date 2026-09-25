import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_tevg_loader

sys.path.append("..")


def check_tevg():
    from util import ROOT

    loader = get_tevg_loader(
        path=os.path.join(ROOT, "tevg"),
        batch_size=1,
        patch_shape=(512, 512),
        fold=1,
        split="train",
        download=True,
    )

    check_loader(loader, 8, instance_labels=False, rgb=True)


if __name__ == "__main__":
    check_tevg()
