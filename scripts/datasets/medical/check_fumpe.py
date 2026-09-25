import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.fumpe import get_fumpe_loader


sys.path.append("..")


def check_fumpe():
    from util import ROOT

    loader = get_fumpe_loader(
        path=os.path.join(ROOT, "fumpe"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_fumpe()
