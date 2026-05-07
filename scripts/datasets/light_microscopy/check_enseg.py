import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_enseg_loader


sys.path.append("..")


def check_enseg():
    from util import ROOT

    loader = get_enseg_loader(
        path=os.path.join(ROOT, "enseg"),
        batch_size=1,
        patch_shape=(512, 512),
        download=True,
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_enseg()
