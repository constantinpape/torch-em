import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_cellpose_loader

sys.path.append("..")


def check_cellpose():
    from util import ROOT

    loader = get_cellpose_loader(
        path=os.path.join(ROOT, "cellpose"),
        split="train",
        patch_shape=(512, 512),
        batch_size=1,
        choice=None,
    )
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_cellpose()
