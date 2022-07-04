import os
import pickle
from concurrent import futures
from glob import glob

import numpy as np
from tqdm import tqdm

from .prepare_shallow2deep import _apply_filters, _get_filters


def visualize_pretrained_rfs(checkpoint, raw, n_forests,
                             sample_random=False, filter_config=None, n_threads=1):
    import napari

    rf_folder = os.path.join(checkpoint, "rfs")
    assert os.path.exists(rf_folder), rf_folder
    rf_paths = glob(os.path.join(rf_folder, "*.pkl"))
    rf_paths.sort()
    if sample_random:
        rf_paths = np.random.choice(rf_paths, size=n_forests)
    else:
        rf_paths = rf_paths[::(len(rf_paths) // n_forests)][:n_forests]

    print("Compute features for input of shape", raw.shape)
    filter_config = _get_filters(raw.ndim, filter_config)
    features = _apply_filters(raw, filter_config)

    def predict_rf(rf_path):
        with open(rf_path, "rb") as f:
            rf = pickle.load(f)
        pred = rf.predict_proba(features)
        pred = pred.reshape(raw.shape + (pred.shape[1],))
        pred = np.moveaxis(pred, -1, 0)
        assert pred.shape[1:] == raw.shape
        return pred

    with futures.ThreadPoolExecutor(n_threads) as tp:
        preds = list(tqdm(tp.map(predict_rf, rf_paths), desc="Predict RFs", total=len(rf_paths)))

    print("Start viewer")
    v = napari.Viewer()
    for path, pred in zip(rf_paths, preds):
        name = os.path.basename(path)
        v.add_image(pred, name=name)
    v.add_image(raw)
    v.grid.enabled = True
    napari.run()
