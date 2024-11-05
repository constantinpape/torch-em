import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_psfhs_loader


sys.path.append("..")


def check_psfhs():
    from util import ROOT

    loader = get_psfhs_loader(
        path=os.path.join(ROOT, "psfhs"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_psfhs()
