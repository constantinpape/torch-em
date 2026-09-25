import os
import sys

from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.cetus import get_cetus_loader


sys.path.append("..")


def check_cetus():
    from util import ROOT

    loader = get_cetus_loader(
        path=os.path.join(ROOT, "cetus"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        ndim=2,
        resize_inputs=True,
        download=True,
        sampler=MinInstanceSampler(),
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_cetus()
