import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.light_microscopy.bbbc024 import get_bbbc024_loader


sys.path.append("..")


def check_bbbc024():
    from util import ROOT

    loader = get_bbbc024_loader(
        path=os.path.join(ROOT, "bbbc024"),
        batch_size=1,
        patch_shape=(32, 256, 256),
        clustering=0,
        snr="high",
        download=True,
    )

    check_loader(loader, 8, instance_labels=True, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_bbbc024()
