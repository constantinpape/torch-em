import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.pansegdata import get_pansegdata_loader


sys.path.append("..")


def check_pansegdata():
    from util import ROOT

    loader = get_pansegdata_loader(
        path=os.path.join(ROOT, "pansegdata"),
        batch_size=1,
        patch_shape=(1, 512, 512),
        modality="t2",
        ndim=2,
        sampler=MinInstanceSampler(),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_pansegdata()
