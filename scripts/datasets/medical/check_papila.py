import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_papila_loader


sys.path.append("..")


def check_papila():
    from util import ROOT

    loader = get_papila_loader(
        path=os.path.join(ROOT, "papila"),
        patch_shape=(512, 512),
        batch_size=2,
        split="test",
        resize_inputs=True,
        task="disc",
        download=True,
    )

    check_loader(loader, 8, rgb=True)


if __name__ == "__main__":
    check_papila()
