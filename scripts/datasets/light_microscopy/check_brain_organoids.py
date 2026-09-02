import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_brain_organoids_loader


sys.path.append("..")


def check_brain_organoids():
    from util import ROOT

    loader = get_brain_organoids_loader(
        path=os.path.join(ROOT, "brain_organoids"),
        patch_shape=(512, 512),
        batch_size=2,
        download=False,
    )

    check_loader(loader, 8, rgb=True)


if __name__ == "__main__":
    check_brain_organoids()
