import os
import sys

from torch_em.data import MinForegroundSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.topcow import get_topcow_loader


sys.path.append("..")


def check_topcow():
    from util import ROOT

    loader = get_topcow_loader(
        path=os.path.join(ROOT, "topcow"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        modality="ct",
        ndim=2,
        resize_inputs=True,
        download=True,
        sampler=MinForegroundSampler(min_fraction=0.001),
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_topcow()
