import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_kvasir_loader


sys.path.append("..")


def check_kvasir():
    # from util import ROOT
    ROOT = "/scratch/share/cidas/cca/data"

    loader = get_kvasir_loader(
        path=os.path.join(ROOT, "kvasir"),
        patch_shape=(512, 512),
        batch_size=2,
        download=True,
        resize_inputs=False,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_kvasir()
