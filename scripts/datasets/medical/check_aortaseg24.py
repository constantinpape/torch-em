import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.aortaseg24 import get_aortaseg24_loader


sys.path.append("..")


def check_aortaseg24():
    from util import ROOT

    loader = get_aortaseg24_loader(
        path=os.path.join(ROOT, "aortaseg24"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        ndim=2,
        download=False,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_aortaseg24()
