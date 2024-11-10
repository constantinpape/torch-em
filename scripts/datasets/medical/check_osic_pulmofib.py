import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_osic_pulmofib_loader


sys.path.append("..")


def check_osic_pulmofib():
    # from util import ROOT
    ROOT = "/scratch/share/cidas/cca/tmp"

    loader = get_osic_pulmofib_loader(
        path=os.path.join(ROOT, "osic_pulmofib"),
        patch_shape=(4, 512, 512),
        ndim=3,
        batch_size=2,
        split="train",
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./osic_pulmofib.png")


if __name__ == "__main__":
    check_osic_pulmofib()
