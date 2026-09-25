import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_precise_loader


sys.path.append("..")


def check_precise():
    from util import ROOT

    loader = get_precise_loader(
        path=os.path.join(ROOT, "precise"),
        batch_size=1,
        patch_shape=(512, 512),
        stain="he",
        n_cases=2,
        level=1,
        download=True,
    )

    check_loader(loader, 8, instance_labels=False, rgb=True, plt=True, save_path="precise.png")


if __name__ == "__main__":
    check_precise()
