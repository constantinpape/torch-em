import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_scaffold_a549_loader


sys.path.append("..")


def check_scaffold_a549():
    from util import ROOT

    loader = get_scaffold_a549_loader(
        path=os.path.join(ROOT, "scaffold_a549"),
        batch_size=1,
        patch_shape=(32, 256, 256),
        split="test",
        download=True,
    )

    check_loader(loader, 4, instance_labels=True, plt=True, save_path="scaffold_a549.png")


if __name__ == "__main__":
    check_scaffold_a549()
