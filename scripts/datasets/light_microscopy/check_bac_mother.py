import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_bac_mother_loader


sys.path.append("..")


def check_bac_mother():
    # from util import ROOT
    ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"

    loader = get_bac_mother_loader(
        path=os.path.join(ROOT, "bac_mother"),
        batch_size=2,
        split="train",
        patch_shape=(512, 512),
        download=True,
        sampler=MinInstanceSampler(),
    )
    check_loader(loader, 8, plt=True, instance_labels=True, save_path="./bac_mother.png")


if __name__ == "__main__":
    check_bac_mother()
