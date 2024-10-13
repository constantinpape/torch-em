import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_hil_toothseg_loader


sys.path.append("..")


def check_hil_toothseg():
    from util import ROOT

    loader = get_hil_toothseg_loader(
        path=os.path.join(ROOT, "hil_toothseg"),
        patch_shape=(512, 512),
        batch_size=2,
        split="train",
        resize_inputs=False,
    )
    check_loader(loader, 8)


if __name__ == "__main__":
    check_hil_toothseg()
