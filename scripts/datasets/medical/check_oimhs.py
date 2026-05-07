import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_oimhs_loader


sys.path.append("..")


def check_oimhs():
    from util import ROOT

    loader = get_oimhs_loader(
        path=os.path.join(ROOT, "oimhs"),
        patch_shape=(512, 512),
        batch_size=2,
        split="test",
        download=True,
        resize_inputs=True,
    )

    check_loader(loader, 8, plt=True, save_path="./oimhs.png")


if __name__ == "__main__":
    check_oimhs()
