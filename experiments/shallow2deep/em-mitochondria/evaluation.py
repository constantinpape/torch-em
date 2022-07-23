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
    path = "/g/kreshuk/pape/Work/data/kasthuri/kasthuri_test.h5"
    with open_file(path, "r") as f:
        # raw = f["raw"][:]
        label = f["labels"][:]
    print(label.shape)
    fg = np.concatenate(
        [(label != 255).all(axis=0)[None]] * label.shape[0],
        axis=0
    )

    # import napari
    # v = napari.Viewer()
    # v.add_labels(label)
    # v.add_labels(fg)
    # napari.run()

    fg = np.where(fg)
    fg_bb = tuple(
        slice(int(gg.min()), int(gg.max()) + 1) for gg in fg
    )
    label = label[fg_bb]
    print(label.shape)


def dice_metric(pred, label, mask=None):
    if pred.ndim == 4:
        pred = pred[0]
    assert pred.shape == label.shape
    # deal with potential ignore label
    if mask is not None:
        pred, label = pred[mask], label[mask]
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
                pred[:, z] = bioimageio.core.predict_with_padding(pp, inp)[0].values[0]
        f_save.create_dataset(model_name, data=pred, compression="gzip")


def require_net_3d(data_path, model_path, model_name, save_path, tiling):
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


def run_evaluation(data_path, save_path, eval_path, label_key="label"):
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            scores = json.load(f)
    else:
        scores = {}

    def load_labels():
        with open_file(data_path, "r") as f:
            labels = f[label_key][:]
        if 255 in labels:
            mask = labels != 255
            print("Have mask!!!!!")
            # getting rid of boundary artifacts
            print("Pix in mask before:", mask.sum())
            mask = np.concatenate(
                [mask.all(axis=0)[None]] * mask.shape[0],
                axis=0
            )
            print("Pix in mask afer:", mask.sum())
        else:
            mask = None
        return labels, mask

    with open_file(save_path, "r") as f:
        missing_names = list(
            set(f.keys()) - set(scores.keys())
        )
        if missing_names:
            labels, mask = load_labels()

        for name in tqdm(missing_names, desc="Run evaluation"):
            pred = f[name][:]
            score = dice_metric(pred, labels, mask)
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


def evaluate_lucchi(version):
    data_path = "/g/kreshuk/pape/Work/data/lucchi/lucchi_test.h5"
    rf_folder = "/g/kreshuk/pape/Work/data/lucchi/ilp3d"
    save_path = f"./bio-models/v{version}/prediction_lucchi.h5"

    rfs = {
        "few-labels": os.path.join(rf_folder, "1.ilp"),
        "medium-labels": os.path.join(rf_folder, "2.ilp"),
        "many-labels": os.path.join(rf_folder, "3.ilp"),
    }
    enhancers_2d, enhancers_anisotropic = get_enhancers(f"./bio-models/v{version}")
    net2d = "./bio-models/v2/DirectModel/MitchondriaEMSegmentation2D.zip"
    net_aniso = "./bio-models/v3/DirectModel/mitochondriaemsegmentationboundarymodel_pytorch_state_dict.zip"

    require_rfs(data_path, rfs, save_path)

    require_enhancers_2d(rfs, enhancers_2d, save_path)
    require_enhancers_3d(rfs, enhancers_anisotropic, save_path)
    # TODO add the 3d enhancers

    require_net_2d(data_path, net2d, "direct_2d", save_path)
    tiling_aniso = {
        "tile": {"z": 32, "y": 256, "x": 256},
        "halo": {"z": 4, "y": 32, "x": 32}
    }
    require_net_3d(data_path, net_aniso, "direct_anisotropic", save_path, tiling_aniso)
    # TODO train and add the 3d network

    eval_path = f"./bio-models/v{version}/lucchi.json"
    scores = run_evaluation(data_path, save_path, eval_path, label_key="labels")
    scores = to_table(scores)
    print("Evaluation results:")
    print(scores.to_markdown(floatfmt=".03f"))


def debug_v4(pred_filter=None):
    import napari
    data_path = "/g/kreshuk/pape/Work/data/kasthuri/kasthuri_test.h5"
    save_path = "./bio-models/v4/prediction_kasthuri.h5"

    print("Load data")
    with open_file(data_path, "r") as f:
        data = f["raw"][:]
        labels = f["labels"][:].astype("uint32")

    with open_file(save_path, "r") as f:
        preds = {}
        for name, ds in tqdm(f.items(), total=len(f)):
            if pred_filter is not None and pred_filter not in name:
                continue
            preds[name] = ds[:]

    print("Start viewer")
    v = napari.Viewer()
    v.add_image(data)
    for name, pred in preds.items():
        v.add_image(pred, name=name)
    v.add_labels(labels)
    napari.run()


if __name__ == "__main__":
    # prepare_eval_v4()

    # debug_v4(pred_filter="few-labels")
    # debug_v4()

    evaluate_lucchi(version=4)
