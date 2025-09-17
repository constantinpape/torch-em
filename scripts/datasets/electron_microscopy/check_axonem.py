import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_axonem_loader


sys.path.append("..")


def check_axonem():
    # from util import ROOT
    ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"

    loader = get_axonem_loader(
        path=os.path.join(ROOT, "axonem"),
        batch_size=2,
        patch_shape=(16, 512, 512),
        download=True,
        sampler=MinInstanceSampler(min_num_instances=3)
    )
    check_loader(loader, 8, instance_labels=True, plt=True, save_path="./axonem.png")


if __name__ == "__main__":
    check_axonem()
