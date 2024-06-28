import os
from glob import glob

import numpy as np


def check_toothfairy():
    root = "/media/anwai/ANWAI/data/toothfairy/ToothFairy_Dataset/Dataset"
    for pid in glob(os.path.join(root, "P*")):
        dense_anns = os.path.join(pid, "gt_alpha.npy")
        if not os.path.exists(dense_anns):
            continue

        inputs = os.path.join(pid, "data.npy")

        image = np.load(inputs)
        gt = np.load(dense_anns)

        import napari
        v = napari.Viewer()
        v.add_image(image)
        v.add_image(gt)
        napari.run()


check_toothfairy()
