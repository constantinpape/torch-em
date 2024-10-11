import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_emneuron_loader

sys.path.append("..")


def check_emneuron():
    # from util import ROOT
    ROOT = "/scratch/share/cidas/cca/data"

    loader = get_emneuron_loader(
        path=os.path.join(ROOT, "emneuron"),
        batch_size=1,
        patch_shape=(8, 512, 512),
        split="val",
    )
    check_loader(loader, 8, instance_labels=True, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_emneuron()
