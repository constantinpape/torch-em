import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_ravir_loader


sys.path.append("..")


def check_ravir():
    from util import ROOT

    loader = get_ravir_loader(
        path=os.path.join(ROOT, "ravir"),
        patch_shape=(256, 256),
        batch_size=2,
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_ravir()
