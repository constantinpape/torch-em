import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_phenocell_loader


sys.path.append("..")


def check_phenocell():
    from util import ROOT

    loader = get_phenocell_loader(
        path=os.path.join(ROOT, "phenocell"),
        patch_shape=(512, 512),
        batch_size=1,
        split="test",
        label_choice="instances",
        download=True,
    )

    check_loader(loader, 8, instance_labels=True, rgb=True, plt=True, save_path="phenocell.png")


if __name__ == "__main__":
    check_phenocell()
