import os
import sys

from torch_em.data.datasets import get_macrophage_a431_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_macrophage_a431():
    from util import ROOT

    loader = get_macrophage_a431_loader(
        path=os.path.join(ROOT, "macrophage_a431"),
        batch_size=1,
        patch_shape=(32, 512, 512),
        names="fib_control",
        organelle="mitochondria",
        download=True,
    )
    check_loader(loader, 8, instance_labels=False)


if __name__ == "__main__":
    check_macrophage_a431()
