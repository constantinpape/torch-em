import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_tsakiroglou_loader


sys.path.append("..")


def check_tsakiroglou():
    from util import ROOT

    for split in ["train", "test"]:
        loader = get_tsakiroglou_loader(
            path=os.path.join(ROOT, "tsakiroglou"),
            patch_shape=(256, 256),
            batch_size=1,
            split=split,
            download=True,
        )
        check_loader(loader, 8, instance_labels=True, rgb=False)


if __name__ == "__main__":
    check_tsakiroglou()
