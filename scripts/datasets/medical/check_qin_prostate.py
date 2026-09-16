import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.qin_prostate import get_qin_prostate_loader


sys.path.append("..")


def check_qin_prostate():
    from util import ROOT

    loader = get_qin_prostate_loader(
        path=os.path.join(ROOT, "qin_prostate"),
        patch_shape=(1, 256, 256),
        batch_size=1,
        sampler=MinInstanceSampler(),
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_qin_prostate()
