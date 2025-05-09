import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_organoid_loader


sys.path.append("..")


def check_organoid():
    # from util import ROOT
    ROOT = "/media/anwai/ANWAI/data"

    loader = get_organoid_loader(
        path=os.path.join(ROOT, "organoid"),
        batch_size=2,
        patch_shape=(512, 512),
        split="train",
        download=True,
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_organoid()
