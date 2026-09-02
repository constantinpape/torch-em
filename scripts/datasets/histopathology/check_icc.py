import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_icc_loader

sys.path.append("..")


def check_icc():
    from util import ROOT

    loader = get_icc_loader(
        path=os.path.join(ROOT, "icc"),
        batch_size=1,
        patch_shape=(512, 512),
        download=True,
    )

    check_loader(loader, 8, instance_labels=True, rgb=True)


if __name__ == "__main__":
    check_icc()
