import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.light_microscopy.bbbc046 import get_bbbc046_loader


sys.path.append("..")


def check_bbbc046():
    from util import ROOT

    loader = get_bbbc046_loader(
        path=os.path.join(ROOT, "bbbc046"), patch_shape=(16, 128, 128), batch_size=1, download=True
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_bbbc046()
