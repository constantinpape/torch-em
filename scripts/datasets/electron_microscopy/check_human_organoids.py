import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_human_organoids_loader


sys.path.append("..")


def check_human_organoids():
    # from util import ROOT
    ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"

    loader = get_human_organoids_loader(
        path=os.path.join(ROOT, "human_organoids"),
        batch_size=2,
        patch_shape=(8, 512, 512),
        organelle="entotic_cell",
        sampler=MinInstanceSampler(),
    )

    check_loader(loader, 8, plt=True, save_path="./human_organoids.png")


if __name__ == "__main__":
    check_human_organoids()
