import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical import get_chaos_loader


sys.path.append("..")


def check_chaos():
    from util import ROOT

    loader = get_chaos_loader(
        path=os.path.join(ROOT, "chaos"),
        patch_shape=(1, 256, 256),
        batch_size=1,
        split="train",
        modality="CT",
        download=True,
        sampler=MinInstanceSampler()
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_chaos()
