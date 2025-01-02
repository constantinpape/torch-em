import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_ifnuclei_loader

sys.path.append("..")


def check_ifnuclei():
    from util import ROOT

    loader = get_ifnuclei_loader(
        path=os.path.join(ROOT, "if_nuclei"),
        patch_shape=(512, 512),
        batch_size=1,
        download=True,
        shuffle=True,
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_ifnuclei()
