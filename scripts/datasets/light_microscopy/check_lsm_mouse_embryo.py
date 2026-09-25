import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_lsm_mouse_embryo_loader


sys.path.append("..")


def check_lsm_mouse_embryo():
    from util import ROOT

    loader = get_lsm_mouse_embryo_loader(
        path=os.path.join(ROOT, "lsm_mouse_embryo"),
        batch_size=1,
        patch_shape=(512, 512),
        split="train",
        task="proliferating_cells",
        download=True,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_lsm_mouse_embryo()
