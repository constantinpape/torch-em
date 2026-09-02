import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_nuclick_loader


sys.path.append("..")


def check_nuclick():
    from util import ROOT

    loader = get_nuclick_loader(
        path=os.path.join(ROOT, "nuclick"),
        patch_shape=(512, 512),
        batch_size=2,
        download=False,
        split="Train",
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_nuclick()
