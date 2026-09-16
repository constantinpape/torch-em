import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_chest_xray_masks_loader


sys.path.append("..")


def check_chest_xray_masks():
    from util import ROOT

    loader = get_chest_xray_masks_loader(
        path=os.path.join(ROOT, "chest_xray_masks"),
        patch_shape=(512, 512),
        batch_size=2,
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./chest_xray_masks.png")


if __name__ == "__main__":
    check_chest_xray_masks()
