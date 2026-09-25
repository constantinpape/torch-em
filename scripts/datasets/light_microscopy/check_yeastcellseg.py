import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_yeastcellseg_loader


sys.path.append("..")


def check_yeastcellseg():
    from util import ROOT

    loader = get_yeastcellseg_loader(
        path=os.path.join(ROOT, "yeastcellseg"),
        batch_size=1,
        patch_shape=(512, 512),
        segmentation_type="instances",
        download=True,
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_yeastcellseg()
