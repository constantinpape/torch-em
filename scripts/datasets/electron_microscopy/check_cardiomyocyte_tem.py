import os
import sys

from torch_em.data.datasets import get_cardiomyocyte_tem_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_cardiomyocyte_tem():
    from util import ROOT

    loader = get_cardiomyocyte_tem_loader(
        path=os.path.join(ROOT, "cardiomyocyte_tem"),
        batch_size=1,
        patch_shape=(512, 512),
        label_choice="mito",
        download=True,
    )
    check_loader(loader, 8)


if __name__ == "__main__":
    check_cardiomyocyte_tem()
