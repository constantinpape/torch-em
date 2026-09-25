import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.tom500 import get_tom500_loader


sys.path.append("..")


def check_tom500():
    from util import ROOT

    loader = get_tom500_loader(
        path=os.path.join(ROOT, "tom500"),
        patch_shape=(1, 512, 512),
        batch_size=2,
        split="train",
        ndim=2,
        download=True,
        sampler=MinInstanceSampler(min_num_instances=3),
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_tom500()
