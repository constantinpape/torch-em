import os
import sys

from torch_em.data.datasets.electron_microscopy import get_cefa_hela_loader
from torch_em.util.debug import check_loader


sys.path.append("..")


def check_cefa_hela():
    from util import ROOT

    loader = get_cefa_hela_loader(
        path=os.path.join(ROOT, "cefa_hela"),
        patch_shape=(512, 512),
        batch_size=1,
        download=True,
    )
    check_loader(loader, 4)


if __name__ == "__main__":
    check_cefa_hela()
