import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.vs_mc_rc import get_vs_mc_rc_loader


sys.path.append("..")


def check_vs_mc_rc():
    from util import ROOT

    loader = get_vs_mc_rc_loader(
        path=os.path.join(ROOT, "vs_mc_rc"),
        patch_shape=(1, 448, 448),
        batch_size=1,
        split="train",
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_vs_mc_rc()
