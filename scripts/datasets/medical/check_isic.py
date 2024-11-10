import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_isic_loader


sys.path.append("..")


def check_isic():
    from util import ROOT

    loader = get_isic_loader(
        path=os.path.join(ROOT, "isic"),
        patch_shape=(512, 512),
        batch_size=2,
        split="test",
        download=True,
        resize_inputs=True,
    )

    check_loader(loader, 8, plt=True, save_path="./isic.png")


if __name__ == "__main__":
    check_isic()
