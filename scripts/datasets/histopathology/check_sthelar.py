import os
import sys

from torch_em.data.datasets import get_sthelar_loader
from torch_em.util.debug import check_loader


sys.path.append("..")


def check_sthelar():
    from util import ROOT

    loader = get_sthelar_loader(
        path=os.path.join(ROOT, "sthelar"),
        batch_size=1,
        patch_shape=(256, 256),
        magnification="20x",
        slides=["heart_s0", "kidney_s0"],
        label_type="instances",
        download=True,
    )
    check_loader(loader, 8, rgb=True, instance_labels=True)


if __name__ == "__main__":
    check_sthelar()
