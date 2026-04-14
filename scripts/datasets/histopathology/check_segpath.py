import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_segpath_loader


sys.path.append("..")


def check_segpath():
    from util import ROOT

    loader = get_segpath_loader(
        path=os.path.join(ROOT, "segpath"),
        patch_shape=(512, 512),
        batch_size=1,
        cell_types="lymphocytes",
        split="train",
        download=True,
    )

    check_loader(loader, 8, instance_labels=False, rgb=True)


if __name__ == "__main__":
    check_segpath()
