import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_ct_cadaiver_loader


sys.path.append("..")


def check_ct_cadaiver():
    from util import ROOT

    loader = get_ct_cadaiver_loader(
        path=os.path.join(ROOT, "ct_cadaiver"),
        patch_shape=(16, 512, 512),
        ndim=3,
        batch_size=1,
        resize_inputs=False,
        download=False,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_ct_cadaiver()
