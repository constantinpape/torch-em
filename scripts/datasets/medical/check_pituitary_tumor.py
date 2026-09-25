import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.sampler import MinForegroundSampler
from torch_em.data.datasets.medical.pituitary_tumor import get_pituitary_tumor_loader


sys.path.append("..")


def check_pituitary_tumor():
    from util import ROOT

    loader = get_pituitary_tumor_loader(
        path=os.path.join(ROOT, "pituitary_tumor"),
        patch_shape=(1, 256, 256),
        batch_size=1,
        ndim=2,
        sampler=MinForegroundSampler(min_fraction=0.001),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_pituitary_tumor()
