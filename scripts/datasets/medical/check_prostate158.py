import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.prostate158 import get_prostate158_loader


sys.path.append("..")


def check_prostate158():
    from util import ROOT

    loader = get_prostate158_loader(
        path=os.path.join(ROOT, "prostate158"),
        patch_shape=(1, 224, 224),
        batch_size=1,
        split="train",
        sequence="t2",
        label_type="anatomy",
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_prostate158()
