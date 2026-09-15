import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.light_microscopy.lungvis import get_lungvis_loader


sys.path.append("..")


def check_lungvis():
    from util import ROOT

    loader = get_lungvis_loader(
        path=os.path.join(ROOT, "lungvis"),
        patch_shape=(32, 512, 512),
        batch_size=1,
        annotation="manual",
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_lungvis()
