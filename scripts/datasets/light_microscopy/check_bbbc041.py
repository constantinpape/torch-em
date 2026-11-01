import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.light_microscopy.bbbc041 import get_bbbc041_loader


sys.path.append("..")


def check_bbbc041():
    from util import ROOT

    loader = get_bbbc041_loader(
        path=os.path.join(ROOT, "bbbc041"),
        patch_shape=(512, 512),
        batch_size=1,
        split="train",
        label_choice="instances",
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_bbbc041()
