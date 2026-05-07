import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_segpc_loader


sys.path.append("..")


def check_segpc():
    from util import ROOT

    loader = get_segpc_loader(
        path=os.path.join(ROOT, "segpc"),
        batch_size=2,
        patch_shape=(512, 512),
        split="validation",
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_segpc()
