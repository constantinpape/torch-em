import os
import sys

from torch_em.data.datasets.electron_microscopy.deepict import get_deepict_actin_data

sys.path.append("..")


def check_deepict_actin():
    from util import ROOT, USE_NAPARI

    data_root = os.path.join(ROOT, "deepict")
    get_deepict_actin_data(data_root, download=True)


if __name__ == "__main__":
    check_deepict_actin()
