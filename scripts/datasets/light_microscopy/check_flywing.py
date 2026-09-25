import os
import sys

from torch_em.data.datasets import get_flywing_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_flywing():
    from util import ROOT

    path = os.path.join(ROOT, "flywing")
    for split in ("train", "val", "test"):
        loader = get_flywing_loader(
            path=path,
            batch_size=1,
            patch_shape=(128, 128),
            split=split,
            download=True,
        )
        check_loader(loader, 3, instance_labels=True)


if __name__ == "__main__":
    check_flywing()
