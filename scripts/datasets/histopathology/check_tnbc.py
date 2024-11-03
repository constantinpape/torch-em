import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_tnbc_loader

sys.path.append("..")


def check_tnbc():
    from util import ROOT

    loader = get_tnbc_loader(
        path=os.path.join(ROOT, "tnbc"),
        patch_shape=(512, 512),
        batch_size=1,
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_tnbc()
