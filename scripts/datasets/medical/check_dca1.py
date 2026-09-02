import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_dca1_loader


sys.path.append("..")


def check_dca1():
    from util import ROOT

    loader = get_dca1_loader(
        path=os.path.join(ROOT, "dca1"),
        patch_shape=(512, 512),
        batch_size=1,
        split="test",
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./dca1.png")


if __name__ == "__main__":
    check_dca1()
