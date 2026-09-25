import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_wing_disc_loader


sys.path.append("..")


def check_wing_disc():
    from util import ROOT

    loader = get_wing_disc_loader(
        path=os.path.join(ROOT, "wing_disc"),
        batch_size=1,
        patch_shape=(32, 256, 256),
        download=True,
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_wing_disc()
