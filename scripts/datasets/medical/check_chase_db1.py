import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.chase_db1 import get_chase_db1_loader


sys.path.append("..")


def check_chase_db1():
    from util import ROOT

    loader = get_chase_db1_loader(
        path=os.path.join(ROOT, "chase_db1"),
        patch_shape=(512, 512),
        batch_size=2,
        split="train",
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, rgb=True, plt=True, save_path="./chase_db1.png")


if __name__ == "__main__":
    check_chase_db1()
