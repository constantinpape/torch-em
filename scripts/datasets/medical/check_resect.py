import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.resect import get_resect_loader


sys.path.append("..")


def check_resect():
    from util import ROOT

    loader = get_resect_loader(
        path=os.path.join(ROOT, "resect"),
        patch_shape=(1, 256, 256),
        batch_size=1,
        source="US",
        phase="before",
        structure="tumor",
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_resect()
