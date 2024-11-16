import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_puma_loader


sys.path.append("..")


def check_puma():
    from util import ROOT

    loader = get_puma_loader(
        path=os.path.join(ROOT, "puma"),
        patch_shape=(512, 512),
        batch_size=2,
        annotations="nuclei",
        download=True,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_puma()
