import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_mitoemv2_loader


sys.path.append("..")


def check_mitoemv2():
    from util import ROOT

    loader = get_mitoemv2_loader(
        path=os.path.join(ROOT, "mitoemv2"),
        batch_size=1,
        patch_shape=(32, 256, 256),
        dataset="stem",
        split="train",
        download=True,
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_mitoemv2()
