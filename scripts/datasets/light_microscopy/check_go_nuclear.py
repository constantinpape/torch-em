import os
import sys

from torch_em.data.datasets import get_gonuclear_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_gonuclear():
    from util import ROOT

    patch_shape = (48, 384, 384)

    loader = get_gonuclear_loader(
        os.path.join(ROOT, "gonuclear"), patch_shape, segmentation_task="nuclei", batch_size=1, download=True
    )
    check_loader(loader, 5, instance_labels=True)

    loader = get_gonuclear_loader(
        os.path.join(ROOT, "gonuclear"), patch_shape, segmentation_task="cells", batch_size=1, download=True
    )
    check_loader(loader, 5, instance_labels=True)


if __name__ == "__main__":
    check_gonuclear()
