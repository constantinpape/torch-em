import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_dsad_loader


sys.path.append("..")


def check_ircadb():
    # from util import ROOT
    ROOT = "/media/anwai/ANWAI/data"

    loader = get_dsad_loader(
        path=os.path.join(ROOT, "dsad"),
        patch_shape=(1, 512, 512),
        split="train",
        batch_size=2,
        download=True,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_ircadb()
