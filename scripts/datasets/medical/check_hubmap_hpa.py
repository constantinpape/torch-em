import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.hubmap_hpa import get_hubmap_hpa_loader


sys.path.append("..")


def check_hubmap_hpa():
    from util import ROOT

    loader = get_hubmap_hpa_loader(
        path=os.path.join(ROOT, "hubmap_hpa"),
        patch_shape=(512, 512),
        batch_size=2,
        download=True,
        resize_inputs=False,
    )

    check_loader(loader, 8, plt=True, save_path="./hubmap_hpa.png")


if __name__ == "__main__":
    check_hubmap_hpa()
