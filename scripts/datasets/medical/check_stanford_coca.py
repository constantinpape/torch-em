import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.stanford_coca import get_stanford_coca_loader


sys.path.append("..")


def check_stanford_coca():
    from util import ROOT

    loader = get_stanford_coca_loader(
        path=os.path.join(ROOT, "stanford_coca"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        sampler=MinInstanceSampler(),
        ndim=2,
        download=False,
    )

    check_loader(loader, 8, plt=True, save_path="./stanford_coca.png")


if __name__ == "__main__":
    check_stanford_coca()
