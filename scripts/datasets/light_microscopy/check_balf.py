import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_balf_loader


sys.path.append("..")


def check_balf():
    from util import ROOT

    loader = get_balf_loader(
        path=os.path.join(ROOT, "balf"),
        batch_size=1,
        patch_shape=(512, 512),
        split="val",
        segmentation_type="instances",
        download=True,
    )

    check_loader(loader, 8, instance_labels=True, rgb=True)


if __name__ == "__main__":
    check_balf()
