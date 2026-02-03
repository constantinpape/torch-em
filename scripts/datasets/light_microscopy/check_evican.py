import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_evican_loader


sys.path.append("..")


def check_evican():
    from util import ROOT

    loader = get_evican_loader(
        path=os.path.join(ROOT, "evican"),
        batch_size=1,
        # patch_shape=(1024, 1024),
        patch_shape=None,
        split="train",
        annotation_type="evican60",
        segmentation_type="nucleus",
        download=True,
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_evican()
