import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from pathlib import Path

import imageio.v3 as imageio

from elf.evaluation import mean_segmentation_accuracy

import torch
from torch_em.util import segmentation, prediction

from micro_sam.evaluation.livecell import _get_livecell_paths

import common


def get_search_ranges(for_min_size=False):
    if for_min_size:
        size_range = range(0, 101, 1)
        size_range = [round(sv) for sv in size_range]
        return size_range
    else:
        thresh_range = np.linspace(0.5, 0.9, 5)  # [0.5. 0.9], freq. - 0.1
        smoothing_range = np.linspace(1, 2, 11)  # [1. 2], freq. - 0.1

        # let's round up the parameters, just in case (to avoid float approximation issues)
        thresh_range = [round(tv, 1) for tv in thresh_range]
        smoothing_range = [round(sv, 1) for sv in smoothing_range]

        return thresh_range, smoothing_range


def get_instance_segmentation(
       gt, cdist, bdist, fg, min_size, center_distance_threshold,
       boundary_distance_threshold, distance_smoothing, for_min_size=False
):
    dm_seg = segmentation.watershed_from_center_and_boundary_distances(
        cdist, bdist, fg, min_size=min_size,
        center_distance_threshold=center_distance_threshold,
        boundary_distance_threshold=boundary_distance_threshold,
        distance_smoothing=distance_smoothing
    )

    # let's evaluate the sample w.r.t. the gt
    instances_msa, instances_sa_acc = mean_segmentation_accuracy(dm_seg, gt, return_accuracies=True)

    # then, store the results w.r.t. the parameter
    if for_min_size:
        res = {
            "min_size": min_size,
            "mSA": instances_msa,
            "SA50": instances_sa_acc[0]
        }
    else:
        res = {
            "center_distance_threshold": center_distance_threshold,
            "boundary_distance_threshold": boundary_distance_threshold,
            "distance_smoothing": distance_smoothing,
            "mSA": instances_msa,
            "SA50": instances_sa_acc[0]
        }
    return res


def do_grid_search(model, device, input_dir, csv_save_dir, for_min_size):
    val_input_paths, val_gt_paths = _get_livecell_paths(input_dir, split="val")
    val_input_paths, val_gt_paths = sorted(val_input_paths), sorted(val_gt_paths)

    for img_path, gt_path in tqdm(zip(val_input_paths, val_gt_paths), total=len(val_input_paths)):
        img = imageio.imread(img_path)  # NOTE: we don't standardize the inputs as we are using the sam ini models
        gt = imageio.imread(gt_path)

        outputs = prediction.predict_with_padding(model, img, device=device, min_divisible=(16, 16))
        fg, cdist, bdist = outputs.squeeze()

        all_combinations = []
        if for_min_size:
            size_range = get_search_ranges(for_min_size)
            for msv in size_range:
                res = get_instance_segmentation(
                    gt, cdist, bdist, fg, min_size=msv,
                    center_distance_threshold=0.5,
                    boundary_distance_threshold=0.6,
                    distance_smoothing=1.0,
                    for_min_size=for_min_size
                )
                all_combinations.append(pd.DataFrame.from_dict([res]))
        else:
            thresh_range, smoothing_range = get_search_ranges()
            for bt in thresh_range:
                for ct in thresh_range:
                    for ds in smoothing_range:
                        res = get_instance_segmentation(
                            gt, cdist, bdist, fg, min_size=25,
                            center_distance_threshold=ct,
                            boundary_distance_threshold=bt,
                            distance_smoothing=ds
                        )
                        all_combinations.append(pd.DataFrame.from_dict([res]))

        df_per_image = pd.concat(all_combinations, ignore_index=True)
        df_per_image.to_csv(
            os.path.join(csv_save_dir, f"{Path(img_path).stem}.csv")
        )


def check_best_scores(
        all_csv_paths, min_size=25, center_distance_threshold=0.5,
        boundary_distance_threshold=0.5, distance_smoothing=1, for_min_size=False
):
    all_msa_per_param, all_sa50_per_param = [], []
    for csv_path in all_csv_paths:
        tmp_df = pd.read_csv(csv_path)

        if for_min_size:
            desired_loc = tmp_df[tmp_df["min_size"] == min_size]
        else:
            desired_loc = tmp_df[
                (tmp_df["center_distance_threshold"] == center_distance_threshold) &
                (tmp_df["boundary_distance_threshold"] == boundary_distance_threshold) &
                (tmp_df["distance_smoothing"] == distance_smoothing)
            ]

        all_msa_per_param.append(desired_loc["mSA"])
        all_sa50_per_param.append(desired_loc["SA50"])

    if for_min_size:
        res_dict = {
            "min_size": min_size,
            "mSA": np.mean(all_msa_per_param),
            "SA50": np.mean(all_sa50_per_param)
        }
    else:
        res_dict = {
            "center_distance_threshold": center_distance_threshold,
            "boundary_distance_threshold": boundary_distance_threshold,
            "distance_smoothing": distance_smoothing,
            "mSA": np.mean(all_msa_per_param),
            "SA50": np.mean(all_sa50_per_param)
        }
    return res_dict


def check_gs_results(res_path, csv_save_dir, for_min_size):
    all_csv_paths = glob(os.path.join(csv_save_dir, "*.csv"))

    all_res = []
    if for_min_size:
        size_range = get_search_ranges(for_min_size)
        for msv in size_range:
            res_dict = check_best_scores(
                all_csv_paths, min_size=msv, for_min_size=True
            )
            all_res.append(pd.DataFrame.from_dict([res_dict]))
    else:
        thresh_range, smoothing_range = get_search_ranges()
        for bt in thresh_range:
            for ct in thresh_range:
                for ds in smoothing_range:
                    res_dict = check_best_scores(
                        all_csv_paths,
                        center_distance_threshold=ct,
                        boundary_distance_threshold=bt,
                        distance_smoothing=ds
                    )
                    all_res.append(pd.DataFrame.from_dict([res_dict]))

    res_df = pd.concat(all_res, ignore_index=True)
    res_df.to_csv(res_path)


def get_best_params(res_path, for_min_size):
    df = pd.read_csv(res_path)
    best_msa_idx = df["mSA"].idxmax()
    best_sa50_idx = df["SA50"].idxmax()

    msa_msg = "The parameters for best mSA is "
    sa50_msg = "The parameters for best SA50 is "

    if for_min_size:
        column = "min_size"
        msa_msg += f"min_size: {df.loc[best_msa_idx][column]}"
        sa50_msg += f"min_size: {df.loc[best_sa50_idx][column]}"
    else:
        columns = ["center_distance_threshold", "boundary_distance_threshold", "distance_smoothing"]
        msa_msg += f"center_distance_threshold: {df.loc[best_msa_idx][columns[0]]} \
            boundary_distance_threshold: {df.loc[best_msa_idx][columns[1]]} \
            distance_smoothing: {df.loc[best_msa_idx][columns[2]]}"
        sa50_msg += f"center_distance_threshold: {df.loc[best_sa50_idx][columns[0]]} \
            boundary_distance_threshold: {df.loc[best_sa50_idx][columns[1]]} \
            distance_smoothing: {df.loc[best_sa50_idx][columns[2]]}"

    print(msa_msg)
    print()
    print(sa50_msg)


def main(gs, for_min_size):
    model_name = "vit_h"
    csv_save_dir = f"/scratch/usr/nimanwai/experiments/unetr/grid_search/{model_name}/ms/"
    os.makedirs(csv_save_dir, exist_ok=True)

    if gs:
        # set the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # let's get the model
        model = common.get_unetr_model(
            model_name=model_name, source_choice="torch-em", patch_shape=(512, 512),
            sam_initialization=True, output_channels=3
        )

        # trained model checkpoint
        model_ckpt = "/scratch/usr/nimanwai/experiments/unetr/try/"  # save_root
        model_ckpt += f"{model_name}/distances/dicebaseddistloss/torch-em-sam/"  # dir structure
        model_ckpt += "checkpoints/livecell-all/best.pt"  # saved checkpoint location

        # let's initialize the model with the trained weights
        model.load_state_dict(torch.load(model_ckpt, map_location=torch.device('cpu'))["model_state"])
        model.to(device)
        model.eval()

        input_dir = "/scratch/usr/nimanwai/data/livecell"
        do_grid_search(model, device, input_dir=input_dir, csv_save_dir=csv_save_dir, for_min_size=for_min_size)

    else:
        res_path = "results" + ("_for-min-size_" if for_min_size else "_") + f"{model_name}_grid_search.csv"
        check_gs_results(res_path, csv_save_dir, for_min_size)
        get_best_params(res_path, for_min_size)


if __name__ == "__main__":
    main(gs=False, for_min_size=True)
