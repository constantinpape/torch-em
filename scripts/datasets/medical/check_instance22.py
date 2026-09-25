import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.instance22 import get_instance22_loader


sys.path.append("..")


def check_instance22():
    from util import ROOT

    loader = get_instance22_loader(
        path=os.path.join(ROOT, "instance22"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        ndim=2,
        download=False,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_instance22()
