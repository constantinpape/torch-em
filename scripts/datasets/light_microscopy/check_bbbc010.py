import os
import sys

from torch_em.data.datasets.light_microscopy.bbbc010 import get_bbbc010_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_bbbc010():
    from util import ROOT

    loader = get_bbbc010_loader(
        path=os.path.join(ROOT, "bbbc010"),
        batch_size=1,
        patch_shape=(512, 512),
        download=True,
    )
    check_loader(loader, 4, instance_labels=True, plt=True, save_path="bbbc010.png")


if __name__ == "__main__":
    check_bbbc010()
