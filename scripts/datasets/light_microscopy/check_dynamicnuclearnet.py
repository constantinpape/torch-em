import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_dynamicnuclearnet_loader


sys.path.append("..")


def check_dynamicnuclearnet():
    from util import ROOT

    loader = get_dynamicnuclearnet_loader(
        path=os.path.join(ROOT, "dynamicnuclearnet"),
        split="train",
        patch_shape=(512, 512),
        batch_size=2,
        download=True,
    )
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_dynamicnuclearnet()
