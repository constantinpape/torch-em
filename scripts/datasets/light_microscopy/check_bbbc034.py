import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_bbbc034_loader


sys.path.append("..")


def check_bbbc034():
    from util import ROOT

    loader = get_bbbc034_loader(
        path=os.path.join(ROOT, "bbbc034"),
        batch_size=1,
        patch_shape=(16, 1024, 1024),
        channel=2,
        download=True,
    )

    check_loader(loader, 4, instance_labels=True, plt=True, save_path="bbbc034.png")


if __name__ == "__main__":
    check_bbbc034()
