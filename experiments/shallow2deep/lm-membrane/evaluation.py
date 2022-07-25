import os
from glob import glob

import bioimageio.core
import numpy as np
import pandas as pd

from elf.io import open_file
from elf.evaluation import dice_score
from ilastik.experimental.api import from_project_file
from tqdm import trange, tqdm
from xarray import DataArray
from torch_em.data.datasets.pnas_arabidopsis import _require_pnas_data

DATA_ROOT = "/scratch/pape/s2d-lm-boundaries"


def prepare_eval_v1():
    _require_pnas_data(os.path.join(DATA_ROOT, "pnas"), download=True)


def require_rfs(data_path, rfs, save_path, raw_key="raw"):
    # check if we need to run any of the predictions
    with open_file(save_path, "a") as f_save:
        if all(name in f_save for name in rfs):
            return

        with open_file(data_path, "r") as f:
            data = f[raw_key][:]
        data = DataArray(data, dims=tuple("zyx"))

        for name, ilp_path in rfs.items():
            if name in f_save:
                continue
            print("Run prediction for ILP", name, ":", ilp_path, "...")
            assert os.path.exists(ilp_path)
            ilp = from_project_file(ilp_path)
            pred = ilp.predict(data).values[..., 1]
            assert pred.shape == data.shape
            f_save.create_dataset(name, data=pred, compression="gzip")


def require_enhancers_2d(rfs, enhancers, save_path):
    with open_file(save_path, "a") as f:
        rf_data = {}
        for enhancer_name, enhancer_path in enhancers.items():
            save_names = [f"{enhancer_name}-{rf_name}" for rf_name in rfs]
            if all(name in f for name in save_names):
                continue
            enhancer = bioimageio.core.load_resource_description(enhancer_path)
            with bioimageio.core.create_prediction_pipeline(enhancer) as pp:
                for rf_name in rfs:
                    save_name = f"{enhancer_name}-{rf_name}"
                    if save_name in f:
                        continue
                    if rf_name not in rf_data:
                        rf_data[rf_name] = f[rf_name][:]
                    rf_pred = rf_data[rf_name]
                    pred = np.zeros((2,) + rf_pred.shape, dtype="float32")
                    for z in trange(rf_pred.shape[0], desc=f"Run prediction for {enhancer_name}-{rf_name}"):
                        inp = DataArray(rf_pred[z][None, None], dims=tuple("bcyx"))
                        predz = bioimageio.core.predict_with_padding(pp, inp)[0].values[0]
                        pred[:, z] = predz
                    f.create_dataset(save_name, data=pred, compression="gzip")


def require_enhancers_3d(rfs, enhancers, save_path):
    tiling = {
        "tile": {"z": 32, "y": 256, "x": 256},
        "halo": {"z": 4, "y": 32, "x": 32}
    }
    with open_file(save_path, "a") as f:
        rf_data = {}
        for enhancer_name, enhancer_path in enhancers.items():
            save_names = [f"{enhancer_name}-{rf_name}" for rf_name in rfs]
            if all(name in f for name in save_names):
                continue
            enhancer = bioimageio.core.load_resource_description(enhancer_path)
            with bioimageio.core.create_prediction_pipeline(enhancer) as pp:
                for rf_name in rfs:
                    save_name = f"{enhancer_name}-{rf_name}"
                    if save_name in f:
                        continue
                    if rf_name not in rf_data:
                        rf_data[rf_name] = f[rf_name][:]
                    rf_pred = rf_data[rf_name]
                    inp = DataArray(rf_pred[None, None], dims=tuple("bczyx"))
                    pred = bioimageio.core.predict_with_tiling(pp, inp, tiling=tiling, verbose=True)[0].values[0]
                    f.create_dataset(save_name, data=pred, compression="gzip")


# TODO finish and run once the correct aniso model has trained
# TODO add plantseg eval
def eval_pnas(version):
    # Trained on one volume from plant0 -> use a few tps from other plants for eval
    rf_folder = "/g/kreshuk/pape/Work/data/pnas/ilps3d"
    data_root = os.path.join(DATA_ROOT, "pnas")
    data_paths = []
    for plant_id in range(1, 5):
        paths = glob(os.path.join(data_root, "*.h5"))
        paths.sort()
        # add first, middle and last timepoint
        data_paths.extend([paths[0], paths[len(paths) // 2], paths[-1]])

    rfs = {
        "few-labels": os.path.join(rf_folder, "1.ilp"),
        "medium-labels": os.path.join(rf_folder, "2.ilp"),
        "many-labels": os.path.join(rf_folder, "3.ilp"),
    }
    enhancer_2d = {"enhancer2d": f"./bio-models/v{version}/s2d-lm-membrane-mouse-embryo_ovules_root-2d-worst_tiles"}
    enhancer_3d = {"enhancer3d:" f"./bio-models/v{version}/s2d-lm-membrane-mouse-embryo_ovules_root-3d-worst_tiles"}
    enhancer_aniso = {"enhancer" "./bio-models/v1/s2d-lm-membrane-covid-if_livecell_mouse-embryo-anisotropic-worst_tiles"}

    save_root = f"./bio-models/v{version}/pnas_predictions"
    os.makedirs(save_root, exist_ok=True)
    for path in data_paths:
        save_path = os.path.join(save_root, os.path.basename(path))
        require_rfs(path, rfs, save_path, raw_key="raw")


if __name__ == "__main__":
    eval_pnas(version=1)
