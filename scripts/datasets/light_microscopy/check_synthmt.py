import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_synthmt_loader


sys.path.append("..")


def check_synthmt():
    from util import ROOT

    loader = get_synthmt_loader(
        path=os.path.join(ROOT, "synthmt"),
        batch_size=1,
        patch_shape=(512, 512),
        download=True,
    )

    check_loader(loader, 8, instance_labels=True, rgb=True)


if __name__ == "__main__":
    check_synthmt()
