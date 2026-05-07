import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_morphonet_loader


sys.path.append("..")


def check_morphonet():
    from util import ROOT

    loader = get_morphonet_loader(
        path=os.path.join(ROOT, "morphonet"),
        batch_size=1,
        patch_shape=(32, 128, 128),
        organism="patiria_miniata",
        download=True,
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_morphonet()
