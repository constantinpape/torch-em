import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_arvidsson_loader

sys.path.append("..")


def check_arvidsson():
    from util import ROOT

    loader = get_arvidsson_loader(
        path=os.path.join(ROOT, "arvidsson"),
        patch_shape=(512, 512),
        batch_size=1,
        split="train",
        download=True,
    )
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_arvidsson()
