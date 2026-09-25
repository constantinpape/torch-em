import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_glas_loader


sys.path.append("..")


def check_glas():
    from util import ROOT

    loader = get_glas_loader(
        path=os.path.join(ROOT, "glas"),
        batch_size=2,
        patch_shape=(512, 512),
        split="train",
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, rgb=True)


if __name__ == "__main__":
    check_glas()
