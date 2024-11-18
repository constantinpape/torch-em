import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_consep_loader


sys.path.append("..")


def check_consep():
    # from util import ROOT
    ROOT = "/media/anwai/ANWAI/data"

    loader = get_consep_loader(
        path=os.path.join(ROOT, "consep"),
        batch_size=2,
        patch_shape=(512, 512),
        download=True,
        split="train",
    )

    check_loader(loader, 8, rgb=True, instance_labels=True)


if __name__ == "__main__":
    check_consep()
