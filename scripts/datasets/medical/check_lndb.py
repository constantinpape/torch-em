import os
import sys

from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.lndb import get_lndb_loader


sys.path.append("..")


def check_lndb():
    from util import ROOT

    loader = get_lndb_loader(
        path=os.path.join(ROOT, "lndb"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        ndim=2,
        consensus_level=1,
        download=True,
        sampler=MinInstanceSampler(),
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_lndb()
