import os

import bioimageio.core
import numpy as np
import pandas as pd
from elf.io import open_file
from elf.evaluation import dice_score
from sklearn.metrics import f1_score
from torch_em.shallow2deep import evaluate_enhancers


# make cut-outs from mito-em for ilastik training and evaluation
def prepare_eval_v1():
    out_folder = "/g/kreshuk/pape/Work/data/mito_em/data/crops"
    os.makedirs(out_folder, exist_ok=True)

    train_bb = np.s_[:50, :1024, :1024]
    test_bb = np.s_[50:, -1024:, -1024:]

    input_path = "/scratch/pape/mito-em/human_val.n5"
    with open_file(input_path, "r") as f:
        dsr = f["raw"]
        dsr.n_threads = 8
        raw_train, raw_test = dsr[train_bb], dsr[test_bb]

        dsl = f["labels"]
        dsl.n_threads = 8
        labels_train, labels_test = dsl[train_bb], dsl[test_bb]

    with open_file(os.path.join(out_folder, "crop_train.h5"), "a") as f:
        f.create_dataset("raw", data=raw_train, compression="gzip")
        f.create_dataset("labels", data=labels_train, compression="gzip")

    with open_file(os.path.join(out_folder, "crop_test.h5"), "a") as f:
        f.create_dataset("raw", data=raw_test, compression="gzip")
        f.create_dataset("labels", data=labels_test, compression="gzip")


def prepare_eval_v2():
    in_path = "/g/kreshuk/data/VNC/data_labeled_mito.h5"
    out_path = "/g/kreshuk/pape/Work/data/isbi/vnc-mitos.h5"
    with open_file(in_path, "r") as f:
        raw = f["raw"][:]
        labels = f["label"][:]
    raw = raw.astype("float32") / 255.0
    with open_file(out_path, "a") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("labels", data=labels, compression="gzip")


def dice_metric(pred, label):
    assert pred.shape[2:] == label.shape
    return dice_score(pred[0, 0], label, threshold_seg=None)


def f1_metric(pred, label):
    assert pred.shape[2:] == label.shape
    return f1_score(label.ravel() > 0, pred[0, 0].ravel() > 0.5)


def _evaluation(
    data_path, rfs, enhancers, rf_channel, save_path, metric=dice_metric, raw_key="raw", label_key="labels", is2d=True
):
    with open_file(data_path, "r") as f:
        raw = f[raw_key][:]
        labels = f[label_key][:]
    if is2d:
        prediction_function = None
    else:
        prediction_function = bioimageio.core.predict_with_tiling
    scores = evaluate_enhancers(
        raw, labels, enhancers, rfs,
        metric=metric, is2d=is2d, rf_channel=rf_channel, save_path=save_path,
        prediction_function=prediction_function
    )
    return scores


def _direct_evaluation(data_path, model_path, save_path, raw_key="raw", label_key="labels", metric=dice_metric):
    import bioimageio.core
    import xarray
    from tqdm import trange

    model = bioimageio.core.load_resource_description(model_path)
    with open_file(data_path, "r") as f:
        raw, labels = f[raw_key][:], f[label_key][:]
    scores = []

    save_key = "direct_predictions"
    with open_file(save_path, "a") as f:
        if save_key in f:
            pred = f[save_key][:]
        else:
            with bioimageio.core.create_prediction_pipeline(model) as pp:
                pred = []
                for z in trange(raw.shape[0]):
                    inp = xarray.DataArray(raw[z][None, None], dims=tuple("bcyx"))
                    predz = pp(inp)[0].values
                    pred.append(predz[None])
                pred = np.concatenate(pred)
                f.create_dataset(save_key, data=pred, compression="gzip")

    for z in range(raw.shape[0]):
        scores.append(metric(pred[z], labels[z]))

    return np.mean(scores)


def _direct_evaluation3d(data_path, model_path, save_path, raw_key="raw", label_key="labels", metric=dice_metric):
    import xarray

    model = bioimageio.core.load_resource_description(model_path)
    with open_file(data_path, "r") as f:
        raw, labels = f[raw_key][:], f[label_key][:]

    save_key = "direct_predictions"
    with open_file(save_path, "a") as f:
        if save_key in f:
            pred = f[save_key][:]
        else:
            with bioimageio.core.create_prediction_pipeline(model) as pp:
                inp = xarray.DataArray(raw[None, None], dims=tuple("bczyx"))
                pred = bioimageio.core.predict_with_tiling(pp, inp, verbose=True)
                pred = pred[0].values
            f.create_dataset(save_key, data=pred, compression="gzip")

    score = metric(pred, labels)
    return score


def evaluation_v1():
    data_root = "/g/kreshuk/pape/Work/data/mito_em/data/crops"
    data_path = os.path.join(data_root, "crop_test.h5")
    rfs = {
        "few-labels": os.path.join(data_root, "rfs", "rf1.ilp"),
        "many-labels": os.path.join(data_root, "rfs", "rf3.ilp"),
    }
    enhancers = {
        "vanilla-enhancer": "./bio-models/v1/EnhancerMitochondriaEM2D/EnhancerMitochondriaEM2D.zip",
        "advanced-enhancer": "./bio-models/v1/EnhancerMitochondriaEM2D-advanced-traing/EnhancerMitochondriaEM2D.zip",
    }
    save_path = "./bio-models/v1/prediction.h5"
    scores = _evaluation(data_path, rfs, enhancers, rf_channel=1, save_path=save_path)

    model_path = "./bio-models/v1/DirectModel/mitchondriaemsegmentation2d_pytorch_state_dict.zip"
    score_raw = _direct_evaluation(data_path, model_path, save_path)

    enhancers = {
        "direct-net": "./bio-models/v1/DirectModel/mitchondriaemsegmentation2d_pytorch_state_dict.zip",
    }
    save_path = "./bio-models/v2/prediction-direct.h5"
    scores_direct = _evaluation(data_path, rfs, enhancers, rf_channel=0, save_path=save_path)
    scores = scores.append(scores_direct.iloc[0])

    print("Evaluation results:")
    print(scores.to_markdown())
    print("Raw net evaluation:", score_raw)


def evaluation_v2():
    data_path = "/g/kreshuk/pape/Work/data/isbi/vnc-mitos.h5"
    rf_folder = "/g/kreshuk/pape/Work/data/vnc/ilps"
    rfs = {
        "few-labels": os.path.join(rf_folder, "vnc-mito1.ilp"),
        "medium-labels": os.path.join(rf_folder, "vnc-mito3.ilp"),
        "many-labels": os.path.join(rf_folder, "vnc-mito6.ilp"),
    }
    enhancers = {
        "vanilla-enhancer": "./bio-models/v2/EnhancerMitochondriaEM2D/EnhancerMitochondriaEM2D.zip",
        "advanced-enhancer": "./bio-models/v2/EnhancerMitochondriaEM2D-advanced-traing/EnhancerMitochondriaEM2D.zip",
    }
    save_path = "./bio-models/v2/prediction.h5"
    scores = _evaluation(data_path, rfs, enhancers, rf_channel=1, save_path=save_path)

    model_path = "./bio-models/v2/DirectModel/MitchondriaEMSegmentation2D.zip"
    score_raw = _direct_evaluation(data_path, model_path, save_path)

    enhancers = {
        "direct-net": "./bio-models/v2/DirectModel/MitchondriaEMSegmentation2D.zip",
    }
    save_path = "./bio-models/v2/prediction-direct.h5"
    scores_direct = _evaluation(data_path, rfs, enhancers, rf_channel=0, save_path=save_path)
    scores = scores.append(scores_direct.iloc[0])

    print("Evaluation results:")
    print(scores.to_markdown())
    print("Raw net evaluation:", score_raw)


def evaluation_v3():
    data_path = "/g/kreshuk/pape/Work/data/isbi/vnc-mitos.h5"
    rf_folder = "/g/kreshuk/pape/Work/data/vnc/ilps3d"

    rfs = {
        "few-labels": os.path.join(rf_folder, "vnc-mito1.ilp"),
        "medium-labels": os.path.join(rf_folder, "vnc-mito2.ilp"),
        "many-labels": os.path.join(rf_folder, "vnc-mito3.ilp"),
    }
    enhancers = {
        "vanilla-enhancer": "./bio-models/v3/EnhancerMitochondriaEM3D/EnhancerMitochondriaEM3D.zip",
        "advanced-enhancer": "./bio-models/v3/EnhancerMitochondriaEM3D-advanced-traing/EnhancerMitochondriaEM3D.zip",
    }
    save_path = "./bio-models/v3/prediction.h5"
    scores = _evaluation(data_path, rfs, enhancers, rf_channel=1, save_path=save_path, is2d=False)

    model_path = "./bio-models/v3/DirectModel/mitochondriaemsegmentationboundarymodel_pytorch_state_dict.zip"
    score_raw = _direct_evaluation3d(data_path, model_path, save_path)

    print("Evaluation results:")
    print(scores.to_markdown())
    print("Raw net evaluation:", score_raw)


def get_enhancers(root):
    names = os.listdir(root)
    enhancers_2d, enhancers_anisotropic = {}, {}
    for name in names:
        parts = name.split("-")
        dim = parts[-2]
        rf = parts[-1]
        path = os.path.join(root, name, f"{name}.zip")
        assert os.path.exists(path)
        if dim == "anisotropic":
            enhancers_anisotropic[f"{dim}-{rf}"] = path
        elif dim == "2d":
            enhancers_2d[f"{dim}-{rf}"] = path
    assert len(enhancers_2d) > 0
    assert len(enhancers_anisotropic) > 0
    return enhancers_2d, enhancers_anisotropic


if __name__ == "__main__":
    # prepare_eval_v1()
    # prepare_eval_v2()

    # evaluation_v1()
    # evaluation_v2()
    evaluation_v3()
