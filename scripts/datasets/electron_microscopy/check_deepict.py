import os
import sys

import h5py
import napari

from torch_em.util.debug import check_loader
from torch_em.data.datasets.electron_microscopy import get_deepict_actin_loader
from torch_em.data import MinForegroundSampler

sys.path.append("..")


def check_deepict_actin_volumes():
    from util import ROOT

    data_root = os.path.join(ROOT, "deepict")

    for dataset in ["00004", "00012"]:
        path = os.path.join(data_root, "deepict_actin", f"{dataset}.h5")
        with h5py.File(path, "r") as f:
            raw = f["raw"][:]
            actin_seg = f["/labels/actin"][:]

        v = napari.Viewer()
        v.add_image(raw)
        v.add_labels(actin_seg)
        v.title = dataset
        napari.run()


def check_deepict_actin_loader():
    from util import ROOT

    data_root = os.path.join(ROOT, "deepict")
    loader = get_deepict_actin_loader(
        data_root, (96, 398, 398), 1, download=True, sampler=MinForegroundSampler(min_fraction=0.025)
    )
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_deepict_actin_volumes()
    # check_deepict_actin_loader()
