import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_yeastms_loader


sys.path.append("..")


def check_yeastms():
    from util import ROOT

    loader = get_yeastms_loader(
        path=os.path.join(ROOT, "yeastms"),
        batch_size=1,
        patch_shape=(256, 256),
        split="train",
        download=True,
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_yeastms()
