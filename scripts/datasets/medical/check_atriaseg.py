import os
import sys

from torch_em.data import MinForegroundSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.atriaseg import get_atriaseg_loader


sys.path.append("..")


def check_atriaseg():
    from util import ROOT

    loader = get_atriaseg_loader(
        path=os.path.join(ROOT, "atriaseg"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        split="train",
        ndim=2,
        resize_inputs=True,
        download=True,
        sampler=MinForegroundSampler(min_fraction=0.001),
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_atriaseg()
