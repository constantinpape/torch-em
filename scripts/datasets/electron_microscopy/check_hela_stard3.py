import os
import sys

from torch_em.data.datasets import get_hela_stard3_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_hela_stard3():
    from util import ROOT

    loader = get_hela_stard3_loader(
        path=os.path.join(ROOT, "hela_stard3"),
        batch_size=1,
        patch_shape=(32, 512, 512),
        organelle="mitochondria",
        volumes="ctrl",
        download=True,
    )
    check_loader(loader, 4, instance_labels=False, plt=True, save_path="./hela_stard3.png")


if __name__ == "__main__":
    check_hela_stard3()
