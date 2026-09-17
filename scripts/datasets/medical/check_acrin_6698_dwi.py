import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.acrin_6698_dwi import get_acrin_6698_dwi_loader


sys.path.append("..")


def check_acrin_6698_dwi():
    from util import ROOT

    loader = get_acrin_6698_dwi_loader(
        path=os.path.join(ROOT, "acrin_6698_dwi"),
        patch_shape=(1, 256, 256),
        batch_size=1,
        sampler=MinInstanceSampler(),
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_acrin_6698_dwi()
