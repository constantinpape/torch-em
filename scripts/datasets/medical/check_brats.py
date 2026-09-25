import os
import sys

from torch_em.data import MinForegroundSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.brats import get_brats_loader


sys.path.append("..")


def check_brats():
    from util import ROOT

    loader = get_brats_loader(
        path=os.path.join(ROOT, "brats"),
        patch_shape=(1, 256, 256),
        batch_size=1,
        modality="t2f",
        region="whole_tumor",
        ndim=2,
        resize_inputs=True,
        download=True,
        sampler=MinForegroundSampler(min_fraction=0.01),
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_brats()
