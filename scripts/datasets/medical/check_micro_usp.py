import os
import sys

from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_micro_usp_loader


sys.path.append("..")


def check_micro_usp():
    from util import ROOT

    loader = get_micro_usp_loader(
        path=os.path.join(ROOT, "micro_usp"),
        patch_shape=(1, 512, 512),
        batch_size=2,
        split="train",
        resize_inputs=True,
        download=True,
        sampler=MinInstanceSampler(),
    )

    check_loader(loader, 8, plt=True, save_path="./micro_usp.png")


if __name__ == "__main__":
    check_micro_usp()
