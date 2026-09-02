import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_aisegcell_loader


sys.path.append("..")


def check_aisegcell():
    from util import ROOT

    loader = get_aisegcell_loader(
        path=os.path.join(ROOT, "aisegcell"),
        patch_shape=(256, 256),
        batch_size=2,
        split="train",
        raw_channel="brightfield",
        download=True,
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_aisegcell()
