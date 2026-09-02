import os
import pickle
from concurrent import futures
from glob import glob
from multiprocessing import cpu_count
from typing import Callable, Dict, Optional, Tuple, Union

import bioimageio.core
import elf.io as io
import numpy as np
import pandas as pd
import xarray
from tqdm import tqdm, trange

from .prepare_shallow2deep import _apply_filters, _get_filters
from .shallow2deep_model import IlastikPredicter


def visualize_pretrained_rfs(
    checkpoint: str,
    raw: np.ndarray,
    n_forests: int,
    sample_random: bool = False,
    filter_config: Optional[Dict] = None,
    n_threads: Optional[int] = None,
) -> None:
    """Visualize pretrained random forests from a shallow2depp checkpoint.

    Args:
        checkpoint: Path to the checkpoint folder.
        raw: The input raw data for prediction.
        n_forests: The number of forests to use for visualization.
        sample_random: Whether to subsample forests randomly or regularly.
        filter_config: The filter configuration.
        n_threads: The number of threads for parallel prediction of forests.
    """
    import napari

    rf_paths = glob(os.path.join(checkpoint, "*.pkl"))
    if len(rf_paths) == 0:
        rf_folder = os.path.join(checkpoint, "rfs")
        assert os.path.exists(rf_folder), rf_folder
        rf_paths = glob(os.path.join(rf_folder, "*.pkl"))
    assert len(rf_paths) > 0
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

    n_threads = cpu_count() if n_threads is None else n_threads
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


def evaluate_enhancers(
    data: np.ndarray,
    labels: np.ndarray,
    enhancers: Dict[str, str],
    ilastik_projects: Dict[str, str],
    metric: Callable,
    prediction_function: Optional[Callable] = None,
    rf_channel: Union[int, Tuple[int, ...]] = 1,
    is2d: bool = False,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """Evaluate enhancers on ilastik random forests from multiple projects.

    Args:
        data: The data for evaluation.
        labels: The labels for evaluation.
        enhancers: Mapping of enhancer names to filepath with enhancer models saved in the biomage.io model format.
        ilastik_projects: Mapping of names to ilastik project paths.
        metric: The metric used for evaluation.
        prediction_function: Function to run prediction with the enhancer.
            By default the bioimageio.prediction pipeline is called directly.
            If given, needs to take the prediction pipeline and data (as xarray) as input.
        rf_channel: The channel(s) of the random forest to be passed as input to the enhancer.
        is2d: Whether to process 3d data as individual slices and average the scores. Is ignored if the data is 2d.
        save_path: Save path for caching the random forest predictions.

    Returns:
        A table with the scores of the enhancers for the different forests and scores of the raw forest predictions.
    """

    assert data.shape == labels.shape
    ndim = data.ndim
    model_ndim = 2 if (data.ndim == 2 or is2d) else 3

    def load_enhancer(enh):
        model = bioimageio.core.load_resource_description(enh)
        return bioimageio.core.create_prediction_pipeline(model)

    # load the enhancers
    models = {name: load_enhancer(enh) for name, enh in enhancers.items()}

    # load the ilps
    ilps = {
        name: IlastikPredicter(path, model_ndim, ilastik_multi_thread=True, output_channel=rf_channel)
        for name, path in ilastik_projects.items()
    }

    def require_rf_prediction(rf, input_, name, axes):
        if save_path is None:
            return rf(input_)
        with io.open_file(save_path, "a") as f:
            if name in f:
                pred = f[name][:]
            else:
                pred = rf(input_)
                # require len(axes) + 2 dimensions (additional batch and channel axis)
                pred = pred[(None,) * (len(axes) + 2 - pred.ndim)]
                assert pred.ndim == len(axes) + 2, f"{pred.ndim}, {len(axes) + 2}"
                f.create_dataset(name, data=pred, compression="gzip")
            return pred

    def require_enh_prediction(enh, rf_pred, name, prediction_function, axes):
        if save_path is None:
            pred = enh(rf_pred) if prediction_function is None else prediction_function(enh, rf_pred)
            pred = pred[0]
            return pred
        with io.open_file(save_path, "a") as f:
            if name in f:
                pred = f[name][:]
            else:
                rf_pred = xarray.DataArray(rf_pred, dims=("b", "c",) + tuple(axes))
                pred = enh(rf_pred) if prediction_function is None else prediction_function(enh, rf_pred)
                pred = pred[0]
                f.create_dataset(name, data=pred, compression="gzip")
            return pred

    def process_chunk(x, y, axes, z=None):
        scores = np.zeros((len(models) + 1, len(ilps)))
        for i, (rf_name, ilp) in enumerate(ilps.items()):
            rf_pred = require_rf_prediction(
                ilp, x,
                rf_name if z is None else f"{rf_name}/{z:04}",
                axes
            )
            for j, (enh_name, enh) in enumerate(models.items()):
                pred = require_enh_prediction(
                    enh, rf_pred,
                    f"{enh_name}/{rf_name}" if z is None else f"{enh_name}/{rf_name}/{z:04}",
                    prediction_function,
                    axes
                )
                score = metric(pred, y)
                scores[j, i] = score
            score = metric(rf_pred, y)
            scores[-1, i] = score

        scores = pd.DataFrame(scores, columns=list(ilps.keys()))
        scores.insert(loc=0, column="enhancer", value=list(models.keys()) + ["rf-score"])
        return scores

    # if we have 2d data, or 3d data that is processed en block,
    # we only have to process a single 'chunk'
    if ndim == 2 or (ndim == 3 and not is2d):
        scores = process_chunk(data, labels, "yx" if ndim == 2 else "zyx")
    elif ndim == 3 and is2d:
        scores = []
        for z in trange(data.shape[0]):
            scores_z = process_chunk(data[z], labels[z], "yx", z)
            scores.append(scores_z)
        scores = pd.concat(scores).groupby("enhancer").mean()
    else:
        raise ValueError("Invalid data dimensions: {ndim}")

    return scores


def load_predictions(save_path: str, n_threads: int = 1) -> Dict[str, np.ndarray]:
    """Load predictions from a save_path created by evaluate_enhancers.

    Args:
        save_path: The path where the predictions were saved.
        n_threads: The number of threads for loading data.

    Returns:
        A mapping of random forest names to the predictions.
    """
    predictions = {}

    def visit(name, node):
        if io.is_group(node):
            return
        node.n_threads = n_threads
        # if we store with 'is2d' individual slices are datasets
        try:
            data_name = "/".join(name.split("/")[:-1])
            z = int(name.split("/")[-1])
            data = node[:]
            pred = predictions.get(data_name, {})
            pred[z] = data
            predictions[data_name] = pred
        # otherwise the above will throw a val error and we just load the array
        except ValueError:
            predictions[name] = node[:]

    with io.open_file(save_path, "r") as f:
        f.visititems(visit)

    def to_vol(pred):
        if isinstance(pred, np.ndarray):
            return pred
        pred = dict(sorted(pred.items()))
        return np.concatenate([pz[None] for pz in pred.values()], axis=0)

    return {name: to_vol(pred) for name, pred in predictions.items()}
