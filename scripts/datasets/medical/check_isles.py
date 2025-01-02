import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinForegroundSampler
from torch_em.data.datasets.medical import get_isles_loader


sys.path.append("..")


def check_isles():
    from util import ROOT

    loader = get_isles_loader(
        path=os.path.join(ROOT, "isles"),
        patch_shape=(1, 112, 112),
        batch_size=2,
        ndim=2,
        modality=None,
        download=True,
        sampler=MinForegroundSampler(min_fraction=0.001),
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_isles()
