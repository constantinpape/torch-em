import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_kits_loader


sys.path.append("..")


def check_kits():
    from util import ROOT

    loader = get_kits_loader(
        path=os.path.join(ROOT, "kits"),
        patch_shape=(16, 512, 512),
        batch_size=2,
        resize_inputs=False,
        download=True,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_kits()
