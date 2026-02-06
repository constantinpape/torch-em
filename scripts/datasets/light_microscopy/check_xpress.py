import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_xpress_loader


sys.path.append("..")


def check_xpress():
    from util import ROOT

    loader = get_xpress_loader(
        path=os.path.join(ROOT, "xpress"),
        batch_size=1,
        patch_shape=(256, 256, 256),
        download=True,
    )

    check_loader(loader, 4, instance_labels=True)


if __name__ == "__main__":
    check_xpress()
