import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_motum_loader


sys.path.append("..")


def check_motum():
    # from util import ROOT
    ROOT = "/media/anwai/ANWAI/data"

    loader = get_motum_loader(
        path=os.path.join(ROOT, "motum"),
        batch_size=1,
        patch_shape=(1, 512, 512),
        ndim=2,
        split="train",
        resize_inputs=True,
        modality="flair",
        download=True,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_motum()
