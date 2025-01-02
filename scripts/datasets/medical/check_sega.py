import os
import sys

from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_sega_loader


sys.path.append("..")


def check_sega():
    from util import ROOT

    loader = get_sega_loader(
        path=os.path.join(ROOT, "sega"),
        patch_shape=(32, 512, 512),
        batch_size=2,
        data_choice="KiTS",
        download=True,
        sampler=MinInstanceSampler(),
    )

    check_loader(loader, 8, plt=True, save_path="./sega.png")


if __name__ == "__main__":
    check_sega()
