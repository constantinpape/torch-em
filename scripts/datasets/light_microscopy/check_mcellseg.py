import os
import sys

from torch_em.data.datasets import get_mcellseg_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_mcellseg():
    from util import ROOT

    loader = get_mcellseg_loader(
        os.path.join(ROOT, "mcellseg"), patch_shape=(512, 512), batch_size=1, download=True,
    )
    check_loader(loader, 8, instance_labels=True, plt=True, save_path="./check_mcellseg.png")


if __name__ == "__main__":
    check_mcellseg()
