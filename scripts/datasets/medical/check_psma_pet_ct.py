import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.psma_pet_ct import get_psma_pet_ct_loader


sys.path.append("..")


def check_psma_pet_ct():
    from util import ROOT

    loader = get_psma_pet_ct_loader(
        path=os.path.join(ROOT, "psma_pet_ct"),
        patch_shape=(1, 200, 200),
        batch_size=1,
        sampler=MinInstanceSampler(),
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_psma_pet_ct()
