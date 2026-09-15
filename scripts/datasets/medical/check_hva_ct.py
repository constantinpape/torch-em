import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.hva_ct import get_hva_ct_loader


sys.path.append("..")


def check_hva_ct():
    from util import ROOT

    loader = get_hva_ct_loader(
        path=os.path.join(ROOT, "hva_ct"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        annotation="vessels",
        msd_path=os.path.join(ROOT, "msd"),
        sampler=MinInstanceSampler(),
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_hva_ct()
