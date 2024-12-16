import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_deepseas_loader


sys.path.append("..")


def check_deepseas():
    # from util import ROOT
    ROOT = "/home/anwai/data"

    loader = get_deepseas_loader(
        path=os.path.join(ROOT, "deepseas"),
        choice="original",
        patch_shape=(512, 512),
        batch_size=2,
        download=True,
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_deepseas()
