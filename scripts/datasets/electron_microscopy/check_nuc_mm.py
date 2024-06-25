import os
import sys

from torch_em.data.datasets import get_nuc_mm_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_nuc_mm():
    from util import ROOT, USE_NAPARI

    nuc_mm_root = os.path.join(ROOT, "nuc_mm")
    if USE_NAPARI:
        use_plt = False
        save_path = None, None
    else:
        use_plt = True
        save_path = "mouse_data.png", "zebrafish_data.png"

    loader = get_nuc_mm_loader(
        nuc_mm_root, "mouse", "train", patch_shape=(1, 192, 192), batch_size=1, download=True
    )
    check_loader(loader, 5, instance_labels=True, plt=use_plt, save_path=save_path[0])

    loader = get_nuc_mm_loader(
        nuc_mm_root, "zebrafish", "train", patch_shape=(1, 64, 64), batch_size=1, download=True
    )
    check_loader(loader, 5, instance_labels=True, plt=use_plt, save_path=save_path[1])


if __name__ == "__main__":
    check_nuc_mm()
