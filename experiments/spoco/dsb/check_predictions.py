import os
from glob import glob

import imageio
import h5py
import numpy as np
from elf.visualisation import simple_grid_view

INPUT_FOLDER = "/g/kreshuk/pape/Work/data/data_science_bowl/dsb2018/test/images"
PRED_FOLDER = "/scratch/pape/dsb/spoco/test"


def check_predictions(checkpoint_name, n_predictions):
    paths = glob(os.path.join(INPUT_FOLDER, "*.tif"))
    np.random.shuffle(paths)

    sampled_paths = [paths[0]]
    images = [imageio.imread(paths[0])]
    shape = images[0].shape

    i, j = 1, 1
    while j < n_predictions:
        image = imageio.imread(paths[i])
        if image.shape == shape:
            sampled_paths.append(paths[i])
            images.append(image)
            j += 1
        i += 1
    assert len(images) == len(sampled_paths)
    image_data = {"images": images}

    segmentations = []
    predictions = []

    seg_name = "hdbscan"
    # seg_name = "mws"
    for path in sampled_paths:
        fname = os.path.splitext(os.path.basename(path))[0]
        pred_path = os.path.join(PRED_FOLDER, f"{fname}.h5")
        with h5py.File(pred_path, "r") as f:
            # pred = f[f"embeddings/{checkpoint_name}"][:]
            # TODO support multichannel
            pred = f[f"embeddings/{checkpoint_name}"][0]
            predictions.append(pred)
            seg = f[f"segmentations/{seg_name}/{checkpoint_name}"][:].astype("uint64")
            segmentations.append(seg)

    image_data["predictions"] = predictions
    label_data = {"segmentation": segmentations}

    simple_grid_view(image_data, label_data)


if __name__ == "__main__":
    check_predictions("dense_embeddings", n_predictions=9)
