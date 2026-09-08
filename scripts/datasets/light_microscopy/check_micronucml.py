import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_micronucml_loader


sys.path.append("..")


def check_micronucml():
    from util import ROOT

    loader = get_micronucml_loader(
        path=os.path.join(ROOT, "micronucml"),
        batch_size=2,
        patch_shape=(224, 224),
        split="train",
        download=True,
        sampler=MinInstanceSampler()
    )
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_micronucml()
