import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_arcade_loader


sys.path.append("..")


def check_arcade():
    # from util import ROOT
    ROOT = "/scratch/share/cidas/cca/data"

    loader = get_arcade_loader(
        path=os.path.join(ROOT, "arcade"),
        split="test",
        patch_shape=(256, 256),
        batch_size=2,
        download=True,
        task="syntax",
        resize_inputs=True,
        sampler=MinInstanceSampler(),
    )
    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_arcade()
