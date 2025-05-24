import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_sa_med2d_loader


sys.path.append("..")


def check_sa_med2d():
    # from util import ROOT
    ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"

    loader = get_sa_med2d_loader(
        path=os.path.join(ROOT, "sa-med2d"),
        patch_shape=(1, 512, 512),
        split="train",
        batch_size=2,
        download=False,
        num_workers=16,
    )

    check_loader(loader, 8, plt=True, save_path="./sa-med2d.png")


if __name__ == "__main__":
    check_sa_med2d()
