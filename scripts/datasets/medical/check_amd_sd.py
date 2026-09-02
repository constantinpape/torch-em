import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_amd_sd_loader


sys.path.append("..")


def check_amd_sd():
    from util import ROOT

    loader = get_amd_sd_loader(
        path=os.path.join(ROOT, "amd_sd"),
        batch_size=2,
        patch_shape=(512, 512),
        split="train",
        download=True,
    )

    check_loader(loader, 8, rgb=True)


if __name__ == "__main__":
    check_amd_sd()
