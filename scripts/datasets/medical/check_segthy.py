import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_segthy_loader


sys.path.append("..")


def check_segthy():
    # from util import ROOT
    ROOT = "/scratch/share/cidas/cca/data"

    loader = get_segthy_loader(
        path=os.path.join(ROOT, "segthy"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        source="MRI",
        ndim=2,
        download=True,
    )
    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_segthy()
