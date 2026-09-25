import os
import sys

from torch_em.data.datasets import get_idr0095_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_idr0095():
    from util import ROOT

    loader = get_idr0095_loader(
        path=os.path.join(ROOT, "idr0095"),
        batch_size=1,
        patch_shape=(512, 512),
        experiment="A",
        download=True,
    )
    check_loader(loader, 4, instance_labels=True, plt=True, save_path="idr0095.png")


if __name__ == "__main__":
    check_idr0095()
