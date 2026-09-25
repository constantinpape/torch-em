import os
import sys

from torch_em.data.datasets import get_fluoem_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_fluoem():
    from util import ROOT

    # bounding box (z_min, z_max, y_min, y_max, x_min, x_max) around a region with proofread
    # axon segmentation, found via WEBKNOSSOS' findData endpoint.
    bounding_box = (1344, 1408, 20992, 21248, 22368, 22624)
    loader = get_fluoem_loader(
        os.path.join(ROOT, "fluoem"), patch_shape=(32, 128, 128), batch_size=1,
        bounding_boxes=[bounding_box], download=True,
    )
    check_loader(loader, 4, instance_labels=True)


if __name__ == "__main__":
    check_fluoem()
