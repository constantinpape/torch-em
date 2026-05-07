import os
import sys

from torch_em.data.datasets import get_malecns_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_malecns():
    from util import ROOT

    loader = get_malecns_loader(
        path=os.path.join(ROOT, "malecns"),
        patch_shape=(32, 256, 256),
        batch_size=1,
        download=True,
    )
    check_loader(loader, 4, instance_labels=True)


if __name__ == "__main__":
    check_malecns()
