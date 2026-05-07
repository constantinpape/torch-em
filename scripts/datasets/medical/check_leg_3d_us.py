import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_leg_3d_us_loader


sys.path.append("..")


def check_leg_3d_us():
    from util import ROOT

    loader = get_leg_3d_us_loader(
        path=os.path.join(ROOT, "leg_3d_us"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        split="train",
        ndim=2,
        download=True,
    )
    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_leg_3d_us()
