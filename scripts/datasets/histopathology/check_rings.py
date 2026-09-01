import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_rings_loader


sys.path.append("..")


def check_rings():
    from util import ROOT

    for split in ["train", "test"]:
        for label_choice in ["glands", "tumor"]:
            loader = get_rings_loader(
                path=os.path.join(ROOT, "rings"),
                patch_shape=(512, 512),
                batch_size=1,
                split=split,
                label_choice=label_choice,
                download=True,
            )
            check_loader(loader, 8, instance_labels=False, rgb=True)


if __name__ == "__main__":
    check_rings()
