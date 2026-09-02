import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_peso_loader

sys.path.append("..")


def check_peso():
    from util import ROOT

    loader = get_peso_loader(
        path=os.path.join(ROOT, "peso"),
        batch_size=1,
        patch_shape=(512, 512),
        resolution_level=2,
        download=True,
    )

    check_loader(loader, 8, instance_labels=False, rgb=True)


if __name__ == "__main__":
    check_peso()
