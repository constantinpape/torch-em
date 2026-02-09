import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_pannuke_loader


sys.path.append("..")


def check_pannuke():
    from util import ROOT

    loader = get_pannuke_loader(
        path=os.path.join(ROOT, "pannuke"),
        batch_size=2,
        patch_shape=(1, 512, 512),
        ndim=2,
        download=True
    )
    check_loader(loader, 8, instance_labels=True, plt=False, rgb=True)


if __name__ == "__main__":
    check_pannuke()
