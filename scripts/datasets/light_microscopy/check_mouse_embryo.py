import os
import sys

from torch_em.data.datasets import get_mouse_embryo_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_mouse_embryo():
    from util import ROOT

    data_root = os.path.join(ROOT, "mouse_embryo")
    loader = get_mouse_embryo_loader(data_root, "nuclei", "train", (8, 512, 512), 1, download=True)
    check_loader(loader, 8, instance_labels=True)

    loader = get_mouse_embryo_loader(data_root, "membrane", "train", (8, 512, 512), 1, download=True)
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_mouse_embryo()
