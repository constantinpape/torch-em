import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_rumc_kidney_loader


sys.path.append("..")


def check_rumc_kidney():
    from util import ROOT

    loader = get_rumc_kidney_loader(
        path=os.path.join(ROOT, "rumc_kidney"),
        batch_size=2,
        patch_shape=(32, 256, 256),
        split="train",
        label_choice="all",
        resize_inputs=False,
        download=True,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_rumc_kidney()
