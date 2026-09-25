import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.remind import get_remind_loader


sys.path.append("..")


def check_remind():
    from util import ROOT

    loader = get_remind_loader(
        path=os.path.join(ROOT, "remind"),
        patch_shape=(1, 256, 256),
        batch_size=1,
        ndim=2,
        sampler=MinInstanceSampler(),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_remind()
