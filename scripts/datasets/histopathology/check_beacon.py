import os
import sys

from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_beacon_loader


sys.path.append("..")


def check_beacon():
    from util import ROOT

    loader = get_beacon_loader(
        path=os.path.join(ROOT, "beacon"),
        batch_size=2,
        patch_shape=(256, 256),
        download=True,
        sampler=MinInstanceSampler(),
    )
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_beacon()
