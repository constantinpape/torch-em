import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.autopet_organ import get_autopet_organ_loader


sys.path.append("..")


def check_autopet_organ():
    from util import ROOT

    loader = get_autopet_organ_loader(
        path=os.path.join(ROOT, "autopet_organ"),
        patch_shape=(1, 400, 400),
        batch_size=1,
        autopet_path=os.path.join(ROOT, "autopet"),
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_autopet_organ()
