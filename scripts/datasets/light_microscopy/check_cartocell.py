import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_cartocell_loader


sys.path.append("..")


def check_cartocell():
    from util import ROOT

    loader = get_cartocell_loader(
        path=os.path.join(ROOT, "cartocell"),
        batch_size=1,
        patch_shape=(1, 512, 512),
        download=True,
    )
    check_loader(loader, 8)


if __name__ == "__main__":
    check_cartocell()
