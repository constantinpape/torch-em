import os
import sys

from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_ircadb_loader


sys.path.append("..")


def check_ircadb():
    # from util import ROOT
    ROOT = "/media/anwai/ANWAI/data"

    loader = get_ircadb_loader(
        path=os.path.join(ROOT, "3d_ircadb"),
        batch_size=2,
        patch_shape=(8, 512, 512),
        label_choice="liver",
        split="train",
        ndim=3,
        download=True,
        resize_inputs=True,
        sampler=MinInstanceSampler(),
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_ircadb()
