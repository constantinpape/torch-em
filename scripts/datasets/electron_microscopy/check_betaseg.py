import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_betaseg_loader


sys.path.append("..")


def check_betaseg():
    # from util import ROOT
    ROOT = "/media/anwai/ANWAI/data"

    loader = get_betaseg_loader(
        path=os.path.join(ROOT, "betaseg"),
        batch_size=2,
        patch_shape=(16, 512, 512),
        download=True,
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_betaseg()
