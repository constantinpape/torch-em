import os
import sys

from torch_em.data.datasets import get_emps_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_emps():
    from util import ROOT

    loader = get_emps_loader(
        path=os.path.join(ROOT, "emps"),
        patch_shape=(512, 512),
        batch_size=1,
        split="train",
        download=True,
        shuffle=True,
    )
    check_loader(loader, 8, instance_labels=True, plt=True, save_path="./emps.png")


if __name__ == "__main__":
    check_emps()
