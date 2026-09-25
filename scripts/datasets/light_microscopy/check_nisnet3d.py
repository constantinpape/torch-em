import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_nisnet3d_loader


sys.path.append("..")


def check_nisnet3d():
    from util import ROOT

    loader = get_nisnet3d_loader(
        path=os.path.join(ROOT, "nisnet3d"),
        batch_size=1,
        patch_shape=(32, 128, 128),
        download=True,
    )

    check_loader(loader, 4, instance_labels=True, plt=True, save_path="nisnet3d.png")


if __name__ == "__main__":
    check_nisnet3d()
