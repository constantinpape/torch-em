from glob import glob

import h5py
import napari
from torch_em.data.datasets.axondeepseg import _require_axondeepseg_data


def check_sem():
    _require_axondeepseg_data("./data", "sem", download=True)
    paths = glob("./data/sem/*.h5")
    for path in paths:
        with h5py.File(path, "r") as f:
            image = f["raw"][:]
            labels = f["labels"][:]
        v = napari.Viewer()
        v.title = path
        v.add_image(image)
        v.add_labels(labels)
        napari.run()


def check_tem():
    _require_axondeepseg_data("./data", "tem", download=True)
    paths = glob("./data/tem/*.h5")
    for path in paths:
        with h5py.File(path, "r") as f:
            image = f["raw"][:]
            labels = f["labels"][:]
        v = napari.Viewer()
        v.title = path
        v.add_image(image)
        v.add_labels(labels)
        napari.run()


check_sem()
# check_tem()
