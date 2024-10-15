import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_cryonuseg_loader


sys.path.append("..")


def check_cryonuseg():
    from util import ROOT

    loader = get_cryonuseg_loader(
        path=os.path.join(ROOT, "cryonuseg"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        rater="b3",
        download=True,
    )
    check_loader(loader, 8, rgb=True, instance_labels=True)


if __name__ == "__main__":
    check_cryonuseg()
