import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_ignite_loader


sys.path.append("..")


def check_ignite():
    from util import ROOT

    loader = get_ignite_loader(
        path=os.path.join(ROOT, "ignite"),
        batch_size=1,
        patch_shape=(512, 512),
        split="train",
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, rgb=True)


if __name__ == "__main__":
    check_ignite()
