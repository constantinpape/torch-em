import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.aeropath import get_aeropath_loader


sys.path.append("..")


def check_aeropath():
    from util import ROOT

    loader = get_aeropath_loader(
        path=os.path.join(ROOT, "aeropath"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        label_choice="airways",
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_aeropath()
