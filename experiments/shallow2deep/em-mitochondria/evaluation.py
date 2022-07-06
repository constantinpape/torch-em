import os
from functools import partial

import numpy as np

from elf.io import open_file
from elf.evaluation import dice_score
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


def _evaluation(data_path, rfs, enhancers, rf_channel, raw_key="raw", label_key="labels"):
    with open_file(data_path, "r") as f:
        raw = f[raw_key][:]
        labels = f[label_key][:]

    metric = partial(dice_score, threshold_seg=None)
    scores = evaluate_enhancers(raw, labels, enhancers, rfs, metric=metric,
                                postprocess_rf=lambda x: x[0, 0],
                                postprocess_enhancer=lambda x: x[0],
                                is2d=True, rf_channel=rf_channel)
    return scores


def _direct_evaluation(data_path, model_path, raw_key="raw", label_key="labels"):
    import bioimageio.core
    import xarray
    from tqdm import trange

    model = bioimageio.core.load_resource_description(model_path)
    with open_file(data_path, "r") as f:
        raw, labels = f[raw_key][:], f[label_key][:]
    scores = []

    with bioimageio.core.create_prediction_pipeline(model) as pp:
        for z in trange(raw.shape[0]):
            inp = xarray.DataArray(raw[z][None, None], dims=tuple("bcyx"))
            pred = pp(inp)[0].values[0, 0]
            scores.append(dice_score(pred, labels[z], threshold_seg=None))

    return np.mean(scores)


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
    scores = _evaluation(data_path, rfs, enhancers, rf_channel=1)
    enhancers = {
        "direct-net": "./bio-models/v1/DirectModel/mitchondriaemsegmentation2d_pytorch_state_dict.zip",
    }
    scores_direct = _evaluation(data_path, rfs, enhancers, rf_channel=0)
    scores = scores.append(scores_direct.iloc[0])

    model_path = "./bio-models/v1/DirectModel/mitchondriaemsegmentation2d_pytorch_state_dict.zip"
    score_raw = _direct_evaluation(data_path, model_path)

    print("Evaluation results:")
    print(scores.to_markdown())
    print("Raw net evaluation:", score_raw)


def evaluation_v2():
    data_path = "/g/kreshuk/data/VNC/data_labeled_mito.h5"
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
    scores = _evaluation(data_path, rfs, enhancers, rf_channel=1, label_key="label")
    enhancers = {
        "direct-net": "./bio-models/v2/DirectModel/MitchondriaEMSegmentation2D.zip",
    }
    scores_direct = _evaluation(data_path, rfs, enhancers, rf_channel=0, label_key="label")
    scores = scores.append(scores_direct.iloc[0])

    model_path = "./bio-models/v2/DirectModel/MitchondriaEMSegmentation2D.zip"
    score_raw = _direct_evaluation(data_path, model_path, label_key="label")

    print("Evaluation results:")
    print(scores.to_markdown())
    print("Raw net evaluation:", score_raw)


if __name__ == "__main__":
    # prepare_eval_v1()
    # evaluation_v1()
    evaluation_v2()
