import json
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


def dice_metric(pred, label, mask=None):
    if pred.ndim == 4:
        pred = pred[0]
    assert pred.shape == label.shape
    # deal with potential ignore label
    if mask is not None:
        pred, label = pred[mask], label[mask]
        assert pred.shape == label.shape
    return dice_score(pred, label, threshold_seg=None)


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


def run_evaluation(data_path, save_path, eval_path, label_key="label"):
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            scores = json.load(f)
    else:
        scores = {}

    def load_labels():
        with open_file(data_path, "r") as f:
            labels = f[label_key][:]
        return labels

    with open_file(save_path, "r") as f:
        missing_names = list(
            set(f.keys()) - set(scores.keys())
        )
        if missing_names:
            labels = load_labels()

        for name in tqdm(missing_names, desc="Run evaluation"):
            pred = f[name][:]
            score = dice_metric(pred, labels)
            scores[name] = float(score)

    with open(eval_path, "w") as f:
        json.dump(scores, f)
    return scores


def to_table(scores):

    # sort the results into enhanncers / rfs with few, medium and many labels
    cols = {"few-labels": {}, "medium-labels": {}, "many-labels": {}}
    for name, score in scores.items():
        for col in cols:
            is_enhancer = False
            if col in name:
                # TODO need to adapt this once we also have a 3d rf
                save_name = "rf3d" if col == name else name.replace(f"-{col}", "")
                cols[col][save_name] = score
                is_enhancer = True
                break
        # direct prediction: don't fit into the categories here, we just put it in the first col (few)
        if not is_enhancer:
            cols["few-labels"][name] = score

    # sort descending after 2d, 3d, anisotropic (alphabetically)
    name_col = list(cols["few-labels"].keys())
    name_col.sort()
    data = []
    for ndim in ("2d", "anisotropic", "3d"):
        for name in name_col:
            if ndim not in name:
                # print("Skipping", ndim, name)
                continue
            row = [name] + [col[name] if name in col else None for col in cols.values()]
            if name == "rf3d":
                data = [row] + data
            else:
                data.append(row)

    df = pd.DataFrame(data, columns=["method"] + list(cols.keys()))
    return df


# TODO add plantseg eval
def eval_pnas(version):
    # Trained on one volume from plant0 -> use a few tps from other plants for eval
    rf_folder = "/g/kreshuk/pape/Work/data/pnas/ilps3d"
    data_root = os.path.join(DATA_ROOT, "pnas")
    data_paths = []
    for plant_id in range(1, 5):
        paths = glob(os.path.join(data_root, f"plant{plant_id}", "*.h5"))
        paths.sort()
        # add first, middle and last timepoint
        data_paths.extend([paths[0], paths[len(paths) // 2], paths[-1]])

    rfs = {
        "few-labels": os.path.join(rf_folder, "1.ilp"),
        "medium-labels": os.path.join(rf_folder, "2.ilp"),
        "many-labels": os.path.join(rf_folder, "3.ilp"),
    }
    # TODO adapt this to the version (different datasets per version)
    enhancer_2d = {
        "enhancer2d": f"./bio-models/v{version}/s2d-lm-membrane-mouse-embryo_ovules_root-2d-worst_tiles/s2d-lm-membrane-mouse-embryo_ovules_root-2d-worst_tiles.zip"
    }
    enhancer_3d = {
        "enhancer3d": f"./bio-models/v{version}/s2d-lm-membrane-mouse-embryo_ovules_root-3d-worst_tiles/s2d-lm-membrane-mouse-embryo_ovules_root-3d-worst_tiles.zip"
    }
    enhancer_aniso = {
        "enhancer_anisotropic": f"./bio-models/v{version}/s2d-lm-membrane-mouse-embryo_ovules_root-anisotropic-worst_tiles/s2d-lm-membrane-mouse-embryo_ovules_root-anisotropic-worst_tiles.zip"
    }

    save_root = f"./bio-models/v{version}/pnas_predictions"
    os.makedirs(save_root, exist_ok=True)
    all_scores = []
    for path in data_paths:
        fname = "_".join(path.split("/")[-2:])
        save_path = os.path.join(save_root, fname)
        print(save_path)
        require_rfs(path, rfs, save_path, raw_key="raw/membrane")
        require_enhancers_2d(rfs, enhancer_2d, save_path)
        require_enhancers_3d(rfs, enhancer_3d, save_path)
        require_enhancers_3d(rfs, enhancer_aniso, save_path)

        eval_path = save_path.replace(".h5", ".json")
        scores = run_evaluation(path, save_path, eval_path, label_key="labels/membrane")
        all_scores.append(scores)

    all_scores = [to_table(score) for score in all_scores]
    scores = pd.concat(all_scores).groupby(level=0).mean()
    scores.insert(loc=0, column="method", value=all_scores[0]["method"])
    print("Evaluation results:")
    print(scores.to_markdown(floatfmt=".03f"))
    print()


# def debug_v1(pred_filter=None):
#     import napari
#     data_path = "/scratch/pape/s2d-lm-boundaries/pnas/"
#     save_path = "./bio-models/v1/pnas_predictions/tp000.h5"
#
#     print("Load data")
#     with open_file(data_path, "r") as f:
#         data = f["raw/cropped"][:]
#         labels = f["labels"][:].astype("uint32")
#
#     with open_file(save_path, "r") as f:
#         preds = {}
#         for name, ds in tqdm(f.items(), total=len(f)):
#             if pred_filter is not None and pred_filter not in name:
#                 continue
#             preds[name] = ds[:]
#
#     print("Start viewer")
#     v = napari.Viewer()
#     v.add_image(data)
#     for name, pred in preds.items():
#         v.add_image(pred, name=name)
#     v.add_labels(labels)
#     napari.run()


if __name__ == "__main__":
    eval_pnas(version=1)
