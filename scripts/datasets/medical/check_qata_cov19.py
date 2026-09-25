import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.qata_cov19 import get_qata_cov19_loader


sys.path.append("..")


def check_qata_cov19():
    from util import ROOT

    loader = get_qata_cov19_loader(
        path=os.path.join(ROOT, "qata_cov19"),
        patch_shape=(224, 224),
        batch_size=1,
        sampler=MinInstanceSampler(),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_qata_cov19()
