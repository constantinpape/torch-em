import os
import sys

from torch_em.data.datasets.electron_microscopy import get_ai4mat_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_ai4mat(name="Super"):
    from util import ROOT

    loader = get_ai4mat_loader(
        path=os.path.join(ROOT, "ai4mat"),
        patch_shape=(512, 512),
        batch_size=2,
        name=name,
        split="train",
        download=True,
        shuffle=True,
    )
    check_loader(loader, 8, instance_labels=False, plt=True, save_path="./ai4mat.png")


if __name__ == "__main__":
    check_ai4mat()
