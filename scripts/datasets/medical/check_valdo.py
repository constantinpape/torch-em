import os
import sys

from torch_em.data import MinForegroundSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.valdo import get_valdo_loader


sys.path.append("..")


def check_valdo():
    from util import ROOT

    loader = get_valdo_loader(
        path=os.path.join(ROOT, "valdo"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        modality="t2s",
        ndim=2,
        resize_inputs=True,
        download=True,
        sampler=MinForegroundSampler(min_fraction=0.00001, p_reject=0.95),
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_valdo()
