import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_deepfucci_loader


sys.path.append("..")


def check_deepfucci():
    from util import ROOT

    for split in ["train", "val"]:
        loader = get_deepfucci_loader(
            path=os.path.join(ROOT, "deepfucci"),
            batch_size=2,
            patch_shape=(256, 256),
            split=split,
            download=True,
        )
        check_loader(loader, 4, instance_labels=True)


if __name__ == "__main__":
    check_deepfucci()
