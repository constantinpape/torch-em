import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.qin_lungct_seg import get_qin_lungct_seg_loader


sys.path.append("..")


def check_qin_lungct_seg():
    from util import ROOT

    loader = get_qin_lungct_seg_loader(
        path=os.path.join(ROOT, "qin_lungct_seg"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        sampler=MinInstanceSampler(),
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_qin_lungct_seg()
