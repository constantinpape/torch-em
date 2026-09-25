import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_lpc_nucseg_loader


sys.path.append("..")


def check_lpc_nucseg():
    from util import ROOT

    loader = get_lpc_nucseg_loader(
        path=os.path.join(ROOT, "lpc_nucseg"),
        batch_size=1,
        # patch_shape=(512, 512),
        patch_shape=None,
        download=True,
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_lpc_nucseg()
