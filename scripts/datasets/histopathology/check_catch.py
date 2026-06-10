import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_catch_loader


sys.path.append("..")


def check_catch():
    from util import ROOT

    # NOTE: This downloads all slides of the chosen tumor type (around 50 whole-slide images).
    loader = get_catch_loader(
        path=os.path.join(ROOT, "catch"),
        patch_shape=(512, 512),
        batch_size=1,
        tumor_types="Histiocytoma",
        download=True,
    )

    check_loader(loader, 8, instance_labels=False, rgb=True)


if __name__ == "__main__":
    check_catch()
