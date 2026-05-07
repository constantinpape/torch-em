import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_mndino_loader


sys.path.append("..")


def check_mndino():
    from util import ROOT

    loader = get_mndino_loader(
        path=os.path.join(ROOT, "mndino"),
        batch_size=2,
        patch_shape=(512, 512),
        split="train",
        label_choice="micronuclei",
        download=True,
        sampler=MinInstanceSampler()
    )
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_mndino()
