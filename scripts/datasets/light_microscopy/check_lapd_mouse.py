import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.light_microscopy.lapd_mouse import get_lapd_mouse_loader


sys.path.append("..")


def check_lapd_mouse():
    from util import ROOT

    loader = get_lapd_mouse_loader(
        path=os.path.join(ROOT, "lapd_mouse"),
        patch_shape=(32, 256, 256),
        batch_size=1,
        resolution="sub4",
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_lapd_mouse()
