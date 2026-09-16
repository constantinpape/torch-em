import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.sliver07 import get_sliver07_loader


sys.path.append("..")


def check_sliver07():
    from util import ROOT

    loader = get_sliver07_loader(
        path=os.path.join(ROOT, "sliver07"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_sliver07()
