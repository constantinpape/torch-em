import os
import sys

from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_cd47_ihcnusc_loader


sys.path.append("..")


def check_cd47_ihcnusc():
    from util import ROOT

    loader = get_cd47_ihcnusc_loader(
        path=os.path.join(ROOT, "cd47_ihcnusc"),
        batch_size=2,
        patch_shape=(512, 512),
        split="train",
        download=True,
        sampler=MinInstanceSampler(),
    )
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_cd47_ihcnusc()
