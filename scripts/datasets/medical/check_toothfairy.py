import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical import get_toothfairy_loader


sys.path.append("..")


def check_toothfairy():
    from util import ROOT

    loader = get_toothfairy_loader(
        path=os.path.join(ROOT, "toothfairy2"),
        patch_shape=(1, 512, 512),
        ndim=2,
        batch_size=2,
        split="train",
        version="v2",
        resize_inputs=False,
        sampler=MinInstanceSampler(),
    )

    check_loader(loader, 8, plt=True, save_path="./toothfairy.png")


if __name__ == "__main__":
    check_toothfairy()
