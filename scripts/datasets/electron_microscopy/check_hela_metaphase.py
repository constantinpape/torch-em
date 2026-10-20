import os
import sys

from torch_em.data.datasets.electron_microscopy import get_hela_metaphase_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_hela_metaphase(name="control"):
    from util import ROOT

    loader = get_hela_metaphase_loader(
        path=os.path.join(ROOT, "hela_metaphase"),
        patch_shape=(32, 512, 512),
        batch_size=1,
        names=name,
        download=True,
        shuffle=True,
    )
    check_loader(loader, 4, instance_labels=False, plt=True, save_path=f"./hela_metaphase_{name}.png")


if __name__ == "__main__":
    check_hela_metaphase()
