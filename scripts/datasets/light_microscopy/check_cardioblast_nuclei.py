import os
import sys

from torch_em.data.datasets import get_cardioblast_nuclei_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_cardioblast_nuclei():
    from util import ROOT

    path = os.path.join(ROOT, "cardioblast_nuclei")
    for split in ("train", "test"):
        loader = get_cardioblast_nuclei_loader(
            path=path,
            batch_size=1,
            patch_shape=(256, 256),
            split=split,
            download=True,
        )
        check_loader(loader, 3, instance_labels=True)


if __name__ == "__main__":
    check_cardioblast_nuclei()
