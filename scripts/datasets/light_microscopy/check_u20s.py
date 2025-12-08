import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_u20s_loader


sys.path.append("..")


def check_u20s():
    from util import ROOT

    loader = get_u20s_loader(
        path=os.path.join(ROOT, "u20s"),
        batch_size=2,
        patch_shape=(512, 512),
        download=True,
    )
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_u20s()
