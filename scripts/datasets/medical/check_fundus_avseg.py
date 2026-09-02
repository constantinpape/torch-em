import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_fundus_avseg_loader


sys.path.append("..")


def check_fundus_avseg():
    from util import ROOT

    loader = get_fundus_avseg_loader(
        path=os.path.join(ROOT, "fundus_avseg"),
        batch_size=2,
        patch_shape=(1024, 1024),
        resize_inputs=True,
        split="train",
        download=True,
    )
    check_loader(loader, 8, rgb=True, instance_labels=True)


if __name__ == "__main__":
    check_fundus_avseg()
