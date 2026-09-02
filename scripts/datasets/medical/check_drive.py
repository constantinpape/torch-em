import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_drive_loader


sys.path.append("..")


def check_drive():
    from util import ROOT

    loader = get_drive_loader(
        path=os.path.join(ROOT, "drive"),
        patch_shape=(512, 512),
        split="train",
        batch_size=1,
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./drive.png")


if __name__ == "__main__":
    check_drive()
