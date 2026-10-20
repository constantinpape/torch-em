import os
import sys

from torch_em.data.datasets.electron_microscopy import get_lewy_mito_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_lewy_mito():
    from util import ROOT

    loader = get_lewy_mito_loader(
        path=os.path.join(ROOT, "lewy_mito"),
        batch_size=1,
        patch_shape=(256, 256),
        groups=None,
        num_workers=0,
        download=True,
        shuffle=True,
    )
    check_loader(loader, 4, instance_labels=False, plt=True, save_path="./lewy_mito.png")


if __name__ == "__main__":
    check_lewy_mito()
