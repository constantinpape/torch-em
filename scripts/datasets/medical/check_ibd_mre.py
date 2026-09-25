import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.ibd_mre import get_ibd_mre_loader


sys.path.append("..")


def check_ibd_mre():
    from util import ROOT

    loader = get_ibd_mre_loader(
        path=os.path.join(ROOT, "ibd_mre"),
        patch_shape=(1, 384, 360),
        batch_size=1,
        sampler=MinInstanceSampler(),
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_ibd_mre()
