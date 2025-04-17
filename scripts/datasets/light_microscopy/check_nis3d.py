import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_nis3d_loader


sys.path.append("..")


def check_nis3d():
    from util import ROOT

    loader = get_nis3d_loader(
        path=os.path.join(ROOT, "nis3d"),
        batch_size=2,
        patch_shape=(16, 512, 512),
        download=True,
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_nis3d()
