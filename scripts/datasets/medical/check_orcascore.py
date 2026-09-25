import os
import sys

from torch_em.data import MinForegroundSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.orcascore import get_orcascore_loader


sys.path.append("..")


def check_orcascore():
    from util import ROOT

    loader = get_orcascore_loader(
        path=os.path.join(ROOT, "orcascore"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        ndim=2,
        resize_inputs=True,
        download=True,
        sampler=MinForegroundSampler(min_fraction=0.00001),
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_orcascore()
