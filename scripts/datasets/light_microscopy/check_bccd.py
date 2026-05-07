import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_bccd_loader


sys.path.append("..")


def check_bccd():
    from util import ROOT

    loader = get_bccd_loader(
        path=os.path.join(ROOT, "bccd"),
        batch_size=1,
        patch_shape=(512, 512),
        split="train",
        download=True,
    )

    check_loader(loader, 8, instance_labels=True, rgb=True)


if __name__ == "__main__":
    check_bccd()
