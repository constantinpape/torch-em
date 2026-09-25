import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_neurosphere_loader


sys.path.append("..")


def check_neurosphere():
    from util import ROOT

    loader = get_neurosphere_loader(
        path=os.path.join(ROOT, "neurosphere"),
        batch_size=1,
        patch_shape=(64, 96, 96),
        download=True,
    )

    check_loader(loader, 4, instance_labels=True, plt=True, save_path="neurosphere.png")


if __name__ == "__main__":
    check_neurosphere()
