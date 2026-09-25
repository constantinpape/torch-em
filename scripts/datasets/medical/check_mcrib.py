import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.mcrib import get_mcrib_loader


sys.path.append("..")


def check_mcrib():
    from util import ROOT

    loader = get_mcrib_loader(
        path=os.path.join(ROOT, "mcrib"),
        patch_shape=(1, 304, 304),
        batch_size=1,
        modality="T2",
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_mcrib()
