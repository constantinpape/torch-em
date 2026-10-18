import os
import sys

from torch_em.data.datasets.electron_microscopy import get_pancreas_fibsem_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_pancreas_fibsem(name="granules"):
    from util import ROOT

    loader = get_pancreas_fibsem_loader(
        path=os.path.join(ROOT, "pancreas_fibsem"),
        patch_shape=(64, 64, 64),
        batch_size=1,
        name=name,
        split="train",
        download=True,
        shuffle=True,
    )
    check_loader(loader, 4, instance_labels=(name == "granules"), plt=True, save_path=f"./pancreas_fibsem_{name}.png")


if __name__ == "__main__":
    check_pancreas_fibsem()
