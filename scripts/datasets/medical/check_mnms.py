import os
import sys

from torch_em.data import MinForegroundSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.mnms import get_mnms_loader


sys.path.append("..")


def check_mnms():
    from util import ROOT

    loader = get_mnms_loader(
        path=os.path.join(ROOT, "mnms"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        split="train",
        ndim=2,
        resize_inputs=True,
        download=True,
        sampler=MinForegroundSampler(min_fraction=0.01),
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_mnms()
