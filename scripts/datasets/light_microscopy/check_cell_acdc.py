import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_cell_acdc_loader


sys.path.append("..")


def check_cell_acdc():
    # from util import ROOT
    ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"

    loader = get_cell_acdc_loader(
        path=os.path.join(ROOT, "cell_acdc"),
        batch_size=2,
        patch_shape=(1, 512, 512),
        download=True,
        sampler=MinInstanceSampler(),
    )
    check_loader(loader, 8, plt=True, instance_labels=True, save_path="./cell_acdc.png")


if __name__ == "__main__":
    check_cell_acdc()
