import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_bmgd_loader


sys.path.append("..")


def check_bmgd():
    from util import ROOT

    loader = get_bmgd_loader(
        path=os.path.join(ROOT, "bmgd"),
        batch_size=1,
        patch_shape=(256, 256),
        stiffness="250pa",
        download=True,
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_bmgd()
