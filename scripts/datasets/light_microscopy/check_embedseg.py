import os
import sys

from torch_em.data.datasets import get_embedseg_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_embedseg():
    from util import ROOT

    names = [
        "Mouse-Organoid-Cells-CBG",
        "Mouse-Skull-Nuclei-CBG",
        "Platynereis-ISH-Nuclei-CBG",
        "Platynereis-Nuclei-CBG",
    ]

    patch_shape = (32, 384, 384)
    for name in names:
        loader = get_embedseg_loader(
            os.path.join(ROOT, "embedseg"), name=name, patch_shape=patch_shape, batch_size=1, download=True
        )
        check_loader(loader, 2, instance_labels=True)


if __name__ == "__main__":
    check_embedseg()
