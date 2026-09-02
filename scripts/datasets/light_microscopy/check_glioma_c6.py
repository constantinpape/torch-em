import os
import sys

from torch_em.data.datasets import get_glioma_c6_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_glioma_c6():
    from util import ROOT

    loader = get_glioma_c6_loader(
        path=os.path.join(ROOT, "glioma_c6"),
        batch_size=1,
        patch_shape=(512, 512),
        subset="spec",
        split="train",
        download=True,
    )
    check_loader(loader, 4, instance_labels=True, plt=True, save_path="glioma_c6.png")


if __name__ == "__main__":
    check_glioma_c6()
