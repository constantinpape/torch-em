import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.histopathology.imc_kidney import get_imc_kidney_loader


sys.path.append("..")


def check_imc_kidney():
    from util import ROOT

    loader = get_imc_kidney_loader(
        path=os.path.join(ROOT, "imc_kidney"),
        patch_shape=(256, 256),
        batch_size=1,
        batches=["Batch1"],
        download=True,
    )

    check_loader(loader, 8, instance_labels=True, rgb=False, plt=True, save_path="check_imc_kidney.png")


if __name__ == "__main__":
    check_imc_kidney()
