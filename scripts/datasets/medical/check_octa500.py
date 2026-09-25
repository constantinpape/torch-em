import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.octa500 import get_octa500_loader


sys.path.append("..")


def check_octa500():
    from util import ROOT

    loader = get_octa500_loader(
        path=os.path.join(ROOT, "octa500"),
        patch_shape=(400, 400),
        batch_size=1,
        subset="6M",
        label_type="large_vessel",
        projection="octa_full",
        download=False,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_octa500()
