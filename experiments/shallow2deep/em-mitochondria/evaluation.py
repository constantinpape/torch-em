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


def prepare_eval_v4():
    import napari
    path = "/g/kreshuk/data/epfl/testing.h5"
    with open_file(path, "r") as f:
        raw = f["raw"][:]
        label = f["label"][:]
    v = napari.Viewer()
    v.add_image(raw)
    v.add_labels(label)
    napari.run()


def dice_metric(pred, label):
    if pred.ndim == 4:
        pred = pred[0]
    assert pred.shape == label.shape
    return dice_score(pred, label, threshold_seg=None)


def require_rfs(data_path, rfs, save_path):
    # check if we need to run any of the predictions
    with open_file(save_path, "a") as f_save:
        if all(name in f_save for name in rfs):
            return

        with open_file(data_path, "r") as f:
            data = f["raw"][:]
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
                        predz = pp(inp)[0].values[0]
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


def require_net_2d(data_path, model_path, model_name, save_path):
    with open_file(save_path, "a") as f_save:
        if model_name in f_save:
            return
        model = bioimageio.core.load_resource_description(model_path)
        with open_file(data_path, "r") as f:
            raw = f["raw"][:]

        pred = np.zeros((2,) + raw.shape, dtype="float32")
        with bioimageio.core.create_prediction_pipeline(model) as pp:
            for z in trange(raw.shape[0], desc=f"Run prediction for model {model_name}"):
                inp = DataArray(raw[z][None, None], dims=tuple("bcyx"))
                pred[:, z] = pp(inp)[0].values[0]
        f_save.create_dataset(model_name, data=pred, compression="gzip")


def require_net_3d(data_path, model_path, model_name, save_path):
    tiling = {
        "tile": {"z": 32, "y": 256, "x": 256},
        "halo": {"z": 4, "y": 32, "x": 32}
    }
    with open_file(save_path, "a") as f_save:
        if model_name in f_save:
            return
        model = bioimageio.core.load_resource_description(model_path)
        with open_file(data_path, "r") as f:
            raw = f["raw"][:]

        pred = np.zeros((2,) + raw.shape, dtype="float32")
        with bioimageio.core.create_prediction_pipeline(model) as pp:
            inp = DataArray(raw[None, None], dims=tuple("bczyx"))
            pred = bioimageio.core.predict_with_tiling(pp, inp, tiling=tiling, verbose=True)[0].values[0]
        f_save.create_dataset(model_name, data=pred, compression="gzip")


def get_enhancers(root):
    names = [os.path.basename(path) for path in glob(os.path.join(root, "s2d-em*"))]
    enhancers_2d, enhancers_anisotropic = {}, {}
    for name in names:
        parts = name.split("-")
        sampling_strategy, dim = parts[-1], parts[-2]
        path = os.path.join(root, name, f"{name}.zip")
        assert os.path.exists(path)
        if dim == "anisotropic":
            enhancers_anisotropic[f"{dim}-{sampling_strategy}"] = path
        elif dim == "2d":
            enhancers_2d[f"{dim}-{sampling_strategy}"] = path
    assert len(enhancers_2d) > 0
    assert len(enhancers_anisotropic) > 0
    return enhancers_2d, enhancers_anisotropic


def run_evaluation(data_path, save_path, eval_path):
    if os.path.exists(eval_path):
        with open(save_path, "r") as f:
            scores = json.load(f)
    else:
        scores = {}

    with open_file(data_path, "r") as f:
        labels = f["label"][:]

    with open_file(save_path, "r") as f:
        for name, ds in tqdm(f.items(), total=len(f), desc="Run evaluation"):
            if name in scores:
                continue
            pred = ds[:]
            score = dice_metric(pred, labels)
            scores[name] = float(score)
    return scores


# TODO
def to_table(scores):
    pass


def evaluation_v4():
    data_path = "/g/kreshuk/pape/Work/data/group_data/epfl/testing.h5"
    rf_folder = "/g/kreshuk/pape/Work/data/epfl/ilastik-projects"
    save_path = "./bio-models/v4/prediction.h5"

    rfs = {
        "few-labels": os.path.join(rf_folder, "2d-1.ilp"),
        "medium-labels": os.path.join(rf_folder, "2d-2.ilp"),
        "many-labels": os.path.join(rf_folder, "2d-3.ilp"),
    }
    require_rfs(data_path, rfs, save_path)

    enhancers_2d, enhancers_anisotropic = get_enhancers("./bio-models/v4")
    require_enhancers_2d(rfs, enhancers_2d, save_path)
    require_enhancers_3d(rfs, enhancers_anisotropic, save_path)

    net2d = "./bio-models/v2/DirectModel/MitchondriaEMSegmentation2D.zip"
    require_net_2d(data_path, net2d, "direct2d", save_path)
    net3d = "./bio-models/v3/DirectModel/mitochondriaemsegmentationboundarymodel_pytorch_state_dict.zip"
    require_net_3d(data_path, net3d, "direct3d", save_path)

    eval_path = "./bio-models/v4/eval.json"
    scores = run_evaluation(data_path, save_path, eval_path)
    scores = to_table(scores)
    print("Evaluation results:")
    print(scores.to_markdown())


def debug_v4():
    import napari
    data_path = "/g/kreshuk/pape/Work/data/group_data/epfl/testing.h5"
    save_path = "./bio-models/v4/prediction.h5"

    print("Load data")
    with open_file(data_path, "r") as f:
        data = f["raw"][:]
        labels = f["label"][:]

    with open_file(save_path, "r") as f:
        preds = {}
        for name, ds in tqdm(f.items(), total=len(f)):
            if ("labels" in name) and ("many" not in name):
                continue
            preds[name] = ds[:]

    print("Start viewer")
    v = napari.Viewer()
    v.add_image(data)
    v.add_labels(labels)
    for name, pred in preds.items():
        v.add_image(pred, name=name)
    napari.run()


if __name__ == "__main__":
    # prepare_eval_v4()
    evaluation_v4()
    # debug_v4()
