import os
import sys

from torch_em.data.datasets import get_neuromast_connectomics_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_neuromast_connectomics():
    from util import ROOT

    # bounding box (z_min, z_max, y_min, y_max, x_min, x_max) around a region with proofread
    # segmentation in sample 'wt1', found via WEBKNOSSOS' findData endpoint.
    bounding_box = (480, 544, 4480, 4736, 4480, 4736)
    loader = get_neuromast_connectomics_loader(
        os.path.join(ROOT, "neuromast_connectomics"), patch_shape=(32, 128, 128), batch_size=1,
        bounding_boxes=[bounding_box], sample="wt1", download=True,
    )
    check_loader(loader, 4, instance_labels=True)


if __name__ == "__main__":
    check_neuromast_connectomics()
