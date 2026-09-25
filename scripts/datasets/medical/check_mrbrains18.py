import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.mrbrains18 import get_mrbrains18_loader


sys.path.append("..")


def check_mrbrains18():
    from util import ROOT

    loader = get_mrbrains18_loader(
        path=os.path.join(ROOT, "mrbrains18"),
        patch_shape=(1, 240, 240),
        batch_size=1,
        split="train",
        modality="t1",
        ndim=2,
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_mrbrains18()
