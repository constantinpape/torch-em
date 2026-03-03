import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_monuseg_loader


sys.path.append("..")


def check_monuseg():
    from util import ROOT

    train_loader = get_monuseg_loader(
        path=os.path.join(ROOT, "monuseg"),
        patch_shape=(512, 512),
        batch_size=2,
        split="train",
        download=True,
        organ_type=["colon", "breast"]
    )
    check_loader(train_loader, 8, instance_labels=True, rgb=True)


if __name__ == "__main__":
    check_monuseg()
