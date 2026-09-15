import os
import sys

from torch_em.data import MinForegroundSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.ribseg import get_ribseg_loader


sys.path.append("..")


def check_ribseg():
    from util import ROOT

    loader = get_ribseg_loader(
        path=os.path.join(ROOT, "ribseg"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        split="val",
        ndim=2,
        download=True,
        sampler=MinForegroundSampler(min_fraction=0.001),
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_ribseg()
