import os
import sys

from torch_em.data.datasets import get_mito_anf_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_mito_anf():
    from util import ROOT

    # bounding box (z_min, z_max, y_min, y_max, x_min, x_max) around a region with proofread
    # mitochondria labels in sample M1, found via WEBKNOSSOS' findData endpoint.
    bounding_box = (558, 686, 7262, 7518, 3412, 3668)
    loader = get_mito_anf_loader(
        os.path.join(ROOT, "mito_anf"), patch_shape=(32, 128, 128), batch_size=1,
        bounding_boxes=[bounding_box], sample="M1", download=True,
    )
    check_loader(loader, 4, instance_labels=True)


if __name__ == "__main__":
    check_mito_anf()
