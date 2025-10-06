import os
import sys

from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_cytodark0_loader


sys.path.append("..")


def check_cytodark0():
    from util import ROOT

    loader = get_cytodark0_loader(
        path=os.path.join(ROOT, "cytodark0"),
        batch_size=2,
        patch_shape=(512, 512),
        split="train",
        download=True,
        sampler=MinInstanceSampler(),
    )
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_cytodark0()
