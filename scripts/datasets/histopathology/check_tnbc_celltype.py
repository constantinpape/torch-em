import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_tnbc_celltype_loader

sys.path.append("..")


def check_tnbc_celltype():
    from util import ROOT

    loader = get_tnbc_celltype_loader(
        path=os.path.join(ROOT, "tnbc_celltype"),
        patch_shape=(512, 512),
        batch_size=1,
        ndim=2,
        download=True,
        split="train",
        label_choice="instances",
    )

    check_loader(loader, 8, instance_labels=True, rgb=True)


if __name__ == "__main__":
    check_tnbc_celltype()
