import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_piccolo_loader


sys.path.append("..")


def check_piccolo():
    from util import ROOT

    loader = get_piccolo_loader(
        path=os.path.join(ROOT, "piccolo"),
        patch_shape=(512, 512),
        batch_size=1,
        split="train",
        resize_inputs=True,
    )

    check_loader(loader, 8, plt=True, save_path="./piccolo.png")


if __name__ == "__main__":
    check_piccolo()
