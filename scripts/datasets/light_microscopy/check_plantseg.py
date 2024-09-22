import os
import sys

from torch_em.data.datasets import get_plantseg_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_plantseg():
    # from util import ROOT
    ROOT = "/media/anwai/ANWAI/data"

    plantseg_root = os.path.join(ROOT, "plantseg")

    loader = get_plantseg_loader(
        plantseg_root, name="root", split="train", patch_shape=(8, 512, 512), batch_size=1, download=True
    )
    check_loader(loader, 8, instance_labels=True)

    loader = get_plantseg_loader(
        plantseg_root, name="ovules", split="train", patch_shape=(8, 512, 512), batch_size=1, download=True
    )
    check_loader(loader, 8, instance_labels=True)

    loader = get_plantseg_loader(
        plantseg_root, name="nuclei", split="train", patch_shape=(8, 512, 512), batch_size=1, download=True
    )
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_plantseg()
