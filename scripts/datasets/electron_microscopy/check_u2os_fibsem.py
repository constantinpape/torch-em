import os
import sys

from torch_em.data.datasets.electron_microscopy import get_u2os_fibsem_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_u2os_fibsem(name="ne"):
    from util import ROOT

    loader = get_u2os_fibsem_loader(
        path=os.path.join(ROOT, "u2os_fibsem"),
        patch_shape=(512, 512),
        batch_size=1,
        name=name,
        download=True,
        shuffle=True,
    )
    check_loader(loader, 4, instance_labels=False, plt=True, save_path=f"./u2os_fibsem_{name}.png")


if __name__ == "__main__":
    check_u2os_fibsem()
