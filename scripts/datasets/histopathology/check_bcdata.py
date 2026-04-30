import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_bcdata_loader


sys.path.append("..")


def check_bcdata():
    from util import ROOT

    loader = get_bcdata_loader(
        path=os.path.join(ROOT, "bcdata"),
        split="train",
        patch_shape=(512, 512),
        batch_size=2,
        cell_radius=2,
        download=True,
    )

    check_loader(loader, 8, instance_labels=False, rgb=True)


if __name__ == "__main__":
    check_bcdata()
