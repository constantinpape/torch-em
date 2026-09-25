import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.cadis import get_cadis_loader


sys.path.append("..")


def check_cadis():
    from util import ROOT

    loader = get_cadis_loader(
        path=os.path.join(ROOT, "cadis"),
        patch_shape=(512, 512),
        batch_size=1,
        split="train",
        ndim=2,
        download=False,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_cadis()
