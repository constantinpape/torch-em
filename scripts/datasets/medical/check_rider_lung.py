import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.rider_lung import get_rider_lung_loader


sys.path.append("..")


def check_rider_lung():
    from util import ROOT

    loader = get_rider_lung_loader(
        path=os.path.join(ROOT, "rider_lung"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        annotation="manual",
        ndim=2,
        sampler=MinInstanceSampler(),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_rider_lung()
