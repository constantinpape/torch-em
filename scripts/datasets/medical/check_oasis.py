import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_oasis_loader


sys.path.append("..")


def check_oasis():
    from util import ROOT

    loader = get_oasis_loader(
        path=os.path.join(ROOT, "oasis"),
        patch_shape=(8, 512, 512),
        split="train",
        batch_size=1,
        label_annotations="4",
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./oasis.png")


if __name__ == "__main__":
    check_oasis()
