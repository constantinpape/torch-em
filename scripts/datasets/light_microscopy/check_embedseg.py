import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_embedseg_loader

sys.path.append("..")


def check_embedseg():
    # from util import ROOT
    ROOT = "/media/anwai/ANWAI/data"

    names = [
        "Mouse-Organoid-Cells-CBG",
        "Mouse-Skull-Nuclei-CBG",
        "Platynereis-ISH-Nuclei-CBG",
        "Platynereis-Nuclei-CBG",
    ]

    for name in names:
        loader = get_embedseg_loader(
            path=os.path.join(ROOT, "embedseg"),
            name=name,
            patch_shape=(32, 384, 384),
            batch_size=2,
            download=True,
        )
        check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_embedseg()
