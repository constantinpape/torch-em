import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_palm_loader


sys.path.append("..")


def check_palm():
    from util import ROOT

    loader = get_palm_loader(
        path=os.path.join(ROOT, "palm"),
        patch_shape=(512, 512),
        batch_size=1,
        split="Training",
        label_choice="disc",
        resize_inputs=True,
        download=True,
        shuffle=True,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_palm()
