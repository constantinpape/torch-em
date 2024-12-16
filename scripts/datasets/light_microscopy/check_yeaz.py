import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_yeaz_loader


sys.path.append("..")


def check_yeaz():
    from util import ROOT

    choice = "phc"  # choose from 'bf' / 'phc'
    if choice == "bf":
        patch_shape, ndim = (512, 512), 2
    else:
        patch_shape, ndim = (1, 512, 512), 3

    loader = get_yeaz_loader(
        path=os.path.join(ROOT, "yeaz"),
        batch_size=2,
        patch_shape=patch_shape,
        choice=choice,
        ndim=ndim,
        download=False,
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_yeaz()
