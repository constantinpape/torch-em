import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.cptac_ucec_tumor import get_cptac_ucec_tumor_loader


sys.path.append("..")


def check_cptac_ucec_tumor():
    from util import ROOT

    loader = get_cptac_ucec_tumor_loader(
        path=os.path.join(ROOT, "cptac_ucec_tumor"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        sampler=MinInstanceSampler(),
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_cptac_ucec_tumor()
