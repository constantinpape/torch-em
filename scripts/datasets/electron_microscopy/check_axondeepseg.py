import os
import sys

from torch_em.data.datasets import get_axondeepseg_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_axondeepseg():
    from util import ROOT, USE_NAPARI

    data_root = os.path.join(ROOT, "axondeepseg")
    if USE_NAPARI:
        use_plt = False
        save_path = None, None
    else:
        use_plt = True
        save_path = "sem_data.png", "tem_data.png"

    loader = get_axondeepseg_loader(
        data_root, name="sem", patch_shape=(1024, 1024), batch_size=1, split="train",
        one_hot_encoding=True, shuffle=True, download=True, val_fraction=0.1
    )
    check_loader(loader, 5, plt=use_plt, save_path=save_path[0])

    loader = get_axondeepseg_loader(
        data_root, name="tem", patch_shape=(1024, 1024), batch_size=1, split="train",
        one_hot_encoding=True, shuffle=True, download=True, val_fraction=0.1
    )
    check_loader(loader, 5, plt=use_plt, save_path=save_path[1])


if __name__ == "__main__":
    check_axondeepseg()
