import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_cytonuke_loader


sys.path.append("..")


def check_cytonuke():
    from util import ROOT

    loader = get_cytonuke_loader(
        path=os.path.join(ROOT, "cytonuke"),
        patch_shape=(256, 256),
        batch_size=2,
        split="train",
        annotations="cell",
        download=True,
    )

    check_loader(loader, 8, instance_labels=True, rgb=True)


if __name__ == "__main__":
    check_cytonuke()
