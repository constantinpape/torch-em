import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_microbeseg_loader


sys.path.append("..")


def check_microbeseg():
    from util import ROOT

    loader = get_microbeseg_loader(
        path=os.path.join(ROOT, "microbeseg"),
        batch_size=1,
        patch_shape=(320, 320),
        split="complete",
        download=True,
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_microbeseg()
