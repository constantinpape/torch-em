import os
import sys

from torch_em.data.datasets import get_mitoem_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_mitoem():
    from util import ROOT

    loader = get_mitoem_loader(os.path.join(ROOT, "mitoem"), splits=["train"], patch_shape=(8, 512, 512),
                               batch_size=1, download=True)
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_mitoem()
