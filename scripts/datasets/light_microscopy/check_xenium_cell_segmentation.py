import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_xenium_cell_segmentation_loader


sys.path.append("..")


def check_xenium_cell_segmentation():
    from util import ROOT

    loader = get_xenium_cell_segmentation_loader(
        path=os.path.join(ROOT, "xenium"),
        patch_shape=(512, 512),
        batch_size=1,
        label_kind="cell",
        raw_channel="all",
        download=False,
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_xenium_cell_segmentation()
