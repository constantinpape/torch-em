import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_nuc_morph_loader


sys.path.append("..")


def check_nuc_morph():
    from util import ROOT

    loader = get_nuc_morph_loader(
        path=os.path.join(ROOT, "nuc_morph"),
        batch_size=1,
        patch_shape=(32, 128, 128),
        split="test",
        download=True,
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_nuc_morph()
