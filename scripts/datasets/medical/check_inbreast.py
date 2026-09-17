import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.inbreast import get_inbreast_loader


sys.path.append("..")


def check_inbreast():
    from util import ROOT

    loader = get_inbreast_loader(
        path=os.path.join(ROOT, "inbreast"),
        patch_shape=(512, 512),
        batch_size=1,
        sampler=MinInstanceSampler(),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./inbreast.png")


if __name__ == "__main__":
    check_inbreast()
