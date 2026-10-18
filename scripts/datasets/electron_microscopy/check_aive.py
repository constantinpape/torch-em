import os
import sys

from torch_em.data.datasets.electron_microscopy import get_aive_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_aive(organelle="mitochondria"):
    from util import ROOT

    loader = get_aive_loader(
        path=os.path.join(ROOT, "aive"),
        patch_shape=(32, 512, 512),
        batch_size=1,
        organelle=organelle,
        download=True,
        shuffle=True,
    )
    check_loader(loader, 4, instance_labels=False, plt=True, save_path=f"./aive_{organelle}.png")


if __name__ == "__main__":
    check_aive()
