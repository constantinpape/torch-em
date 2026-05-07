import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_montgomery_loader


sys.path.append("..")


def check_montgomery():
    from util import ROOT

    loader = get_montgomery_loader(
        path=os.path.join(ROOT, "montgomery"),
        patch_shape=(512, 512),
        batch_size=2,
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./montgomery.png")


if __name__ == "__main__":
    check_montgomery()
