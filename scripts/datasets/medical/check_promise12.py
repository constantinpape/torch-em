import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.promise12 import get_promise12_loader


sys.path.append("..")


def check_promise12():
    from util import ROOT

    loader = get_promise12_loader(
        path=os.path.join(ROOT, "promise12"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_promise12()
