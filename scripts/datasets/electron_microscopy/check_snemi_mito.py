import os
import sys

from torch_em.data.datasets.electron_microscopy import get_snemi_mito_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_snemi_mito():
    from util import ROOT

    loader = get_snemi_mito_loader(
        path=os.path.join(ROOT, "snemi_mito"),
        batch_size=1,
        patch_shape=(512, 512),
        split="train",
        num_workers=0,
        download=True,
        shuffle=True,
    )
    check_loader(loader, 4, instance_labels=False, plt=True, save_path="./snemi_mito.png")


if __name__ == "__main__":
    check_snemi_mito()
