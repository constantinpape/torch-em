import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_medussa_loader


sys.path.append("..")


def check_medussa():
    from util import ROOT

    loader = get_medussa_loader(
        path=os.path.join(ROOT, "medussa"),
        batch_size=1,
        patch_shape=(512, 512),
        split="train",
        download=True,
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_medussa()
