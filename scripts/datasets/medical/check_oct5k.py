import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.oct5k import get_oct5k_loader


sys.path.append("..")


def check_oct5k():
    from util import ROOT

    loader = get_oct5k_loader(
        path=os.path.join(ROOT, "oct5k"),
        patch_shape=(512, 512),
        batch_size=2,
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./oct5k.png")


if __name__ == "__main__":
    check_oct5k()
