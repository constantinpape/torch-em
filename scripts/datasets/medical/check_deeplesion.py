import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.deeplesion import get_deeplesion_loader


sys.path.append("..")


def check_deeplesion():
    from util import ROOT

    loader = get_deeplesion_loader(
        path=os.path.join(ROOT, "deeplesion"),
        patch_shape=(1, 256, 256),
        batch_size=1,
        category=None,
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_deeplesion()
