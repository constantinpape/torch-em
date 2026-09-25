import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_cisd_loader


sys.path.append("..")


def check_cisd():
    from util import ROOT

    loader = get_cisd_loader(
        path=os.path.join(ROOT, "cisd"),
        batch_size=1,
        patch_shape=(256, 256),
        mode="center_slice",
        download=True,
    )

    check_loader(loader, 4, instance_labels=True, plt=True, save_path="cisd.png")


if __name__ == "__main__":
    check_cisd()
