import os
import sys

from torch_em.data.datasets.electron_microscopy import get_etchacell_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_etchacell():
    from util import ROOT

    loader = get_etchacell_loader(
        path=os.path.join(ROOT, "etchacell"),
        batch_size=1,
        patch_shape=(32, 512, 512),
        rois=["ROI_1716-7800-517"],
        num_workers=0,
        download=True,
        shuffle=True,
    )
    check_loader(loader, 4, instance_labels=False, plt=True, save_path="./etchacell.png")


if __name__ == "__main__":
    check_etchacell()
