import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.light_microscopy.bbbc050 import get_bbbc050_loader


sys.path.append("..")


def check_bbbc050():
    from util import ROOT

    loader = get_bbbc050_loader(
        path=os.path.join(ROOT, "bbbc050"),
        patch_shape=(32, 112, 112),
        batch_size=1,
        split="train",
        label_type="QCANet",
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_bbbc050()
