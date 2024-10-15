import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_lgg_mri_loader


sys.path.append("..")


def check_lgg_mri():
    from util import ROOT

    loader = get_lgg_mri_loader(
        path=os.path.join(ROOT, "lgg_mri"),
        patch_shape=(8, 512, 512),
        batch_size=1,
        download=True,
    )
    check_loader(loader, 8)


if __name__ == "__main__":
    check_lgg_mri()
