import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.light_microscopy.bbbc032 import get_bbbc032_loader


sys.path.append("..")


def check_bbbc032():
    from util import ROOT

    loader = get_bbbc032_loader(
        path=os.path.join(ROOT, "bbbc032"),
        batch_size=1,
        patch_shape=(32, 512, 512),
        channel=3,
        download=True,
    )

    check_loader(loader, 8, instance_labels=True, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_bbbc032()
