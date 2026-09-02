import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_panoptils_loader


sys.path.append("..")


def check_panoptils():
    from util import ROOT

    loader = get_panoptils_loader(
        path=os.path.join(ROOT, "panoptils"),
        batch_size=1,
        patch_shape=(512, 512),
        label_choice="instances",
        download=True,
    )

    check_loader(loader, 8, instance_labels=True, rgb=True)


if __name__ == "__main__":
    check_panoptils()
