import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_us_nerve_loader


sys.path.append("..")


def check_us_nerve():
    from util import ROOT

    loader = get_us_nerve_loader(
        path=os.path.join(ROOT, "us_nerve"),
        patch_shape=(512, 512),
        batch_size=2,
        download=True,
        resize_inputs=False,
    )

    check_loader(loader, 8, plt=True, save_path="./us_nerve.png")


if __name__ == "__main__":
    check_us_nerve()
