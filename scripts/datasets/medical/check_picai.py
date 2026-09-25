import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.picai import get_picai_loader


sys.path.append("..")


def check_picai():
    from util import ROOT

    loader = get_picai_loader(
        path=os.path.join(ROOT, "picai"),
        patch_shape=(1, 640, 640),
        batch_size=1,
        annotation="whole_gland",
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_picai()
