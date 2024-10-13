import os
import sys

from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_mbh_seg_loader


sys.path.append("..")


def check_mbh_seg():
    from util import ROOT

    loader = get_mbh_seg_loader(
        path=os.path.join(ROOT, "mbh_seg"),
        patch_shape=(1, 512, 512),
        ndim=2,
        batch_size=2,
        resize_inputs=False,
        download=True,
        sampler=MinInstanceSampler(),
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_mbh_seg()
