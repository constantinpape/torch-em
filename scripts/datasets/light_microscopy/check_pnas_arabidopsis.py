import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_pnas_arabidopsis_loader


sys.path.append("..")


def check_pnas_arabidopsis():
    from util import ROOT

    loader = get_pnas_arabidopsis_loader(
        path=os.path.join(ROOT, "pnas_arabidopsis"),
        batch_size=2,
        patch_shape=(16, 512, 512),
        download=True,
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_pnas_arabidopsis()
