import os
import sys

from torch_em.data.datasets import get_kasthuri_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_kasthuri():
    from util import ROOT, USE_NAPARI

    loader = get_kasthuri_loader(os.path.join(ROOT, "kasthuri"), "train", (8, 512, 512), 1, download=True)
    check_loader(loader, 8, instance_labels=True, plt=not USE_NAPARI)


if __name__ == "__main__":
    check_kasthuri()
