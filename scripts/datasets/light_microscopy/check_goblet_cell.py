import os
import sys

from torch_em.data.datasets.light_microscopy.goblet_cell import get_goblet_cell_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_goblet_cell():
    from util import ROOT

    loader = get_goblet_cell_loader(
        path=os.path.join(ROOT, "goblet_cell"),
        batch_size=2,
        patch_shape=(256, 256),
        split="train",
        patched=True,
        download=True,
    )
    check_loader(loader, 4, instance_labels=True, rgb=True, plt=True, save_path="goblet_cell.png")


if __name__ == "__main__":
    check_goblet_cell()
