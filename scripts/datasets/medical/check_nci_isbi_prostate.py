import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.nci_isbi_prostate import get_nci_isbi_prostate_loader


sys.path.append("..")


def check_nci_isbi_prostate():
    from util import ROOT

    loader = get_nci_isbi_prostate_loader(
        path=os.path.join(ROOT, "nci_isbi_prostate"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        split="train",
        ndim=2,
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_nci_isbi_prostate()
