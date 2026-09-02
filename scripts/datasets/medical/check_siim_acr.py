import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical import get_siim_acr_loader


sys.path.append("..")


def check_siim_acr():
    from util import ROOT

    loader = get_siim_acr_loader(
        path=os.path.join(ROOT, "siim_acr"),
        split="train",
        patch_shape=(512, 512),
        batch_size=2,
        download=True,
        resize_inputs=True,
        sampler=MinInstanceSampler(),
    )

    check_loader(loader, 8, plt=True, save_path="./siim_acr.png")


if __name__ == "__main__":
    check_siim_acr()
