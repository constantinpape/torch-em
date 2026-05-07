import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_duke_liver_loader


sys.path.append("..")


def check_duke_liver():
    from util import ROOT

    loader = get_duke_liver_loader(
        path=os.path.join(ROOT, "duke_liver"),
        patch_shape=(32, 512, 512),
        batch_size=2,
        split="train",
        download=False,
    )

    check_loader(loader, 8, plt=True, save_path="./duke_liver.png")


if __name__ == "__main__":
    check_duke_liver()
