import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.retouch import get_retouch_loader


sys.path.append("..")


def check_retouch():
    from util import ROOT

    loader = get_retouch_loader(
        path=os.path.join(ROOT, "retouch"),
        patch_shape=(32, 512, 512),
        batch_size=1,
        sampler=MinInstanceSampler(),
        ndim=3,
        download=False,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_retouch()
