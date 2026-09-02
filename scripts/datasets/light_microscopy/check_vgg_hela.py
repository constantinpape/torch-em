import os
import sys

from torch_em.data.datasets import get_vgg_hela_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_vgg_hela():
    from util import ROOT

    loader = get_vgg_hela_loader(os.path.join(ROOT, "hela-vgg"), "train", (1, 256, 256), 1, download=True)
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_vgg_hela()
