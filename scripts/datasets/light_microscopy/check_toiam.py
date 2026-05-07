import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_toiam_loader


sys.path.append("..")


def check_toiam():
    from util import ROOT

    loader = get_toiam_loader(
        path=os.path.join(ROOT, "toiam"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        download=True,
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_toiam()
