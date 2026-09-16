import os
import sys

from torch_em.data.datasets import get_cortex_connectomics_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_cortex_connectomics():
    from util import ROOT

    # bounding box (z_min, z_max, y_min, y_max, x_min, x_max) around a region with proofread
    # segmentation in sample 'mouse_v2', found via WEBKNOSSOS' findData endpoint.
    bounding_box = (2496, 2560, 5024, 5280, 3424, 3680)
    loader = get_cortex_connectomics_loader(
        os.path.join(ROOT, "cortex_connectomics"), patch_shape=(32, 128, 128), batch_size=1,
        bounding_boxes=[bounding_box], sample="mouse_v2", download=True,
    )
    check_loader(loader, 4, instance_labels=True)


if __name__ == "__main__":
    check_cortex_connectomics()
