import os
import sys

from torch_em.data import MinForegroundSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.isbi_mslesion import get_isbi_mslesion_loader


sys.path.append("..")


def check_isbi_mslesion():
    from util import ROOT

    loader = get_isbi_mslesion_loader(
        path=os.path.join(ROOT, "isbi_mslesion"),
        patch_shape=(1, 224, 224),
        batch_size=1,
        modality="flair",
        rater="both",
        ndim=2,
        resize_inputs=True,
        download=True,
        sampler=MinForegroundSampler(min_fraction=0.001),
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_isbi_mslesion()
