import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.couinaud import get_couinaud_loader


sys.path.append("..")


def check_couinaud():
    from util import ROOT

    loader = get_couinaud_loader(
        path=os.path.join(ROOT, "couinaud"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        annotation="couinaud",
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_couinaud()
