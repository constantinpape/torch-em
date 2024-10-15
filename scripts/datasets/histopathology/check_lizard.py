import os
import sys

from torch_em.data.datasets import get_lizard_loader
from torch_em.util.debug import check_loader


sys.path.append("..")


def check_lizard():
    from util import ROOT

    loader = get_lizard_loader(
        path=os.path.join(ROOT, "lizard"),
        patch_shape=(512, 512),
        batch_size=1,
        download=True
    )
    check_loader(loader, 8, rgb=True, instance_labels=True)


if __name__ == "__main__":
    check_lizard()
