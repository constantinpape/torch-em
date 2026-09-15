import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.insect_anatomy import get_insect_anatomy_loader


sys.path.append("..")


def check_insect_anatomy():
    from util import ROOT

    loader = get_insect_anatomy_loader(
        path=os.path.join(ROOT, "insect_anatomy"),
        patch_shape=(512, 512),
        batch_size=1,
        split="train",
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_insect_anatomy()
