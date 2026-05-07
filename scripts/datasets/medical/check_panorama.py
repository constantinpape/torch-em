import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_panorama_loader


sys.path.append("..")


def check_panorama():
    from util import ROOT

    loader = get_panorama_loader(
        path=os.path.join(ROOT, "panorama"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        ndim=2,
        download=True,
    )
    check_loader(loader, 8)


if __name__ == "__main__":
    check_panorama()
