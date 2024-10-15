import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_jsrt_loader


sys.path.append("..")


def check_jsrt():
    from util import ROOT

    loader = get_jsrt_loader(
        path=os.path.join(ROOT, "jsrt"),
        split="test",
        patch_shape=(256, 256),
        batch_size=2,
        choice=None,
        download=True,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_jsrt()
