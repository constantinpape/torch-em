import os
import sys

from torch_em.data.datasets import get_l4_dense_reconstruction_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_l4_dense_reconstruction():
    from util import ROOT

    # bounding box (z_min, z_max, y_min, y_max, x_min, x_max) around a region with proofread
    # segmentation in the 'full' sample, found via WEBKNOSSOS' findData endpoint.
    bounding_box = (1728, 1792, 4160, 4416, 2720, 2976)
    loader = get_l4_dense_reconstruction_loader(
        os.path.join(ROOT, "l4_dense_reconstruction"), patch_shape=(32, 128, 128), batch_size=1,
        bounding_boxes=[bounding_box], sample="full", download=True,
    )
    check_loader(loader, 4, instance_labels=True)


if __name__ == "__main__":
    check_l4_dense_reconstruction()
