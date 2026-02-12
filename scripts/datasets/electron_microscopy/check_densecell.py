import os
import sys

from torch_em.data.datasets import get_densecell_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_densecell():
    from util import ROOT

    loader = get_densecell_loader(
        path=os.path.join(ROOT, "densecell"),
        split="train",
        # patch_shape=(8, 512, 512),
        patch_shape=None,
        batch_size=1,
        label_choice="cell",
        label_type="instances",
        download=True,
    )
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_densecell()
