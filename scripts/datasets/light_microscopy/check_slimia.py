import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_slimia_loader


sys.path.append("..")


def check_slimia():
    from util import ROOT

    loader = get_slimia_loader(
        path=os.path.join(ROOT, "slimia"),
        batch_size=1,
        patch_shape=(512, 512),
        microscope="OperaPhenix",
        download=True,
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_slimia()
