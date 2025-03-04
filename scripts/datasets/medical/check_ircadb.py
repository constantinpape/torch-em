import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_ircadb_loader


sys.path.append("..")


def check_ircadb():
    # from util import ROOT
    ROOT = "/media/anwai/ANWAI/data"

    loader = get_ircadb_loader(
        path=os.path.join(ROOT, "3d_ircadb"),
        patch_shape=(1, 512, 512),
        split="train",
        label_choice="bone",
        ndim=3,
        batch_size=2,
        download=True,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_ircadb()
