import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinForegroundSampler
from torch_em.data.datasets.medical.wmh import get_wmh_loader


sys.path.append("..")


def check_wmh():
    from util import ROOT

    loader = get_wmh_loader(
        path=os.path.join(ROOT, "wmh"),
        patch_shape=(1, 240, 240),
        batch_size=1,
        split="train",
        modality="FLAIR",
        sampler=MinForegroundSampler(min_fraction=0.001),
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_wmh()
