import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.ctspine1k import get_ctspine1k_loader


sys.path.append("..")


def check_ctspine1k():
    from util import ROOT

    loader = get_ctspine1k_loader(
        path=os.path.join(ROOT, "ctspine1k"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        source="colonog",
        sampler=MinInstanceSampler(),
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_ctspine1k()
