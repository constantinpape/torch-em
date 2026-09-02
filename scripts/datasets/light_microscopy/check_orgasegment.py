import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.light_microscopy import get_orgasegment_loader


sys.path.append("..")


def check_orgasegment():
    from util import ROOT

    loader = get_orgasegment_loader(
        path=os.path.join(ROOT, "orgasegment"),
        split="train",
        patch_shape=(512, 512),
        batch_size=2,
        download=True,
    )
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_orgasegment()
