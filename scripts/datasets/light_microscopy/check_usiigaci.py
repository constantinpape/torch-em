import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_usiigaci_loader


sys.path.append("..")


def check_usiigaci():
    # from util import ROOT
    ROOT = "/scratch/share/cidas/cca/data"

    loader = get_usiigaci_loader(
        path=os.path.join(ROOT, "usiigaci"),
        patch_shape=(512, 512),
        batch_size=1,
        download=True,
        split="train",
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_usiigaci()
