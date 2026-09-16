import os
import sys

from torch_em.data import MinForegroundSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_bonbid_hie_loader


sys.path.append("..")


def check_bonbid_hie():
    from util import ROOT

    loader = get_bonbid_hie_loader(
        path=os.path.join(ROOT, "bonbid_hie"),
        patch_shape=(1, 160, 160),
        batch_size=2,
        split="train",
        modality=None,
        ndim=2,
        download=True,
        sampler=MinForegroundSampler(min_fraction=0.001),
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_bonbid_hie()
