import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.brainptm import get_brainptm_loader


sys.path.append("..")


def check_brainptm():
    from util import ROOT

    loader = get_brainptm_loader(
        path=os.path.join(ROOT, "brainptm"),
        patch_shape=(1, 128, 128),
        batch_size=1,
        tract="OR_left",
        sampler=MinInstanceSampler(),
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_brainptm()
