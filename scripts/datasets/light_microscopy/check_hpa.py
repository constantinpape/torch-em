import os
import sys

from torch_em.data.datasets import get_hpa_segmentation_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_hpa():
    from util import ROOT

    loader = get_hpa_segmentation_loader(
        path=os.path.join(ROOT, "hpa"),
        split="train",
        patch_shape=(1024, 1024),
        batch_size=1,
        channels=["protein"],
        download=True,
    )
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_hpa()
