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


def get_search_ranges():
    thresh_range = np.linspace(0.5, 0.9, 5)  # [0.5. 0.9], freq. - 0.1
    smoothing_range = np.linspace(1, 2, 11)  # [1. 2], freq. - 0.1

    # let's round up the parameters, just in case (to avoid float approximation issues)
    thresh_range = [round(tv, 1) for tv in thresh_range]
    smoothing_range = [round(sv, 1) for sv in smoothing_range]

    return thresh_range, smoothing_range


def do_grid_search(model, device, input_dir, csv_save_dir):
    val_input_paths, val_gt_paths = _get_livecell_paths(input_dir, split="val")
    val_input_paths, val_gt_paths = sorted(val_input_paths), sorted(val_gt_paths)

    for img_path, gt_path in tqdm(zip(val_input_paths, val_gt_paths), total=len(val_input_paths)):
        img = imageio.imread(img_path)  # NOTE: we don't standardize the inputs as we are using the sam ini models
        gt = imageio.imread(gt_path)

        outputs = prediction.predict_with_padding(model, img, device=device, min_divisible=(16, 16))
        fg, cdist, bdist = outputs.squeeze()

        all_combinations = []
        thresh_range, smoothing_range = get_search_ranges()

        for bt in thresh_range:
            for ct in thresh_range:
                for ds in smoothing_range:
                    dm_seg = segmentation.watershed_from_center_and_boundary_distances(
                        cdist, bdist, fg, min_size=25,
                        center_distance_threshold=ct,
                        boundary_distance_threshold=bt,
                        distance_smoothing=ds
                    )

                    # let's evaluate the sample w.r.t. the gt
                    instances_msa, instances_sa_acc = mean_segmentation_accuracy(dm_seg, gt, return_accuracies=True)

                    # then, store the results w.r.t. the parameter
                    res = {
                        "center_distance_threshold": ct,
                        "boundary_distance_threshold": bt,
                        "distance_smoothing": ds,
                        "mSA": instances_msa,
                        "SA50": instances_sa_acc[0]
                    }
                    all_combinations.append(pd.DataFrame.from_dict([res]))

        df_per_image = pd.concat(all_combinations, ignore_index=True)
        df_per_image.to_csv(
            os.path.join(csv_save_dir, f"{Path(img_path).stem}.csv")
        )


def check_gs_results(csv_save_dir):
    all_csv_paths = glob(os.path.join(csv_save_dir, "*.csv"))

    all_res = []
    thresh_range, smoothing_range = get_search_ranges()

    for bt in thresh_range:
        for ct in thresh_range:
            for ds in smoothing_range:
                # here, we now have the combination of bt, ct and ds
                # let's use the combinations and evaluate how they performed for all images
                all_msa_per_param, all_sa50_per_param = [], []
                for csv_path in all_csv_paths:
                    tmp_df = pd.read_csv(csv_path)
                    desired_loc = tmp_df[
                        (tmp_df["center_distance_threshold"] == ct) &
                        (tmp_df["boundary_distance_threshold"] == bt) &
                        (tmp_df["distance_smoothing"] == ds)
                    ]
                    all_msa_per_param.append(desired_loc["mSA"])
                    all_sa50_per_param.append(desired_loc["SA50"])

                res_dict = {
                    "center_distance_threshold": ct,
                    "boundary_distance_threshold": bt,
                    "distance_smoothing": ds,
                    "mSA": np.mean(all_msa_per_param),
                    "SA50": np.mean(all_sa50_per_param)
                }
                all_res.append(pd.DataFrame.from_dict([res_dict]))

    res_df = pd.concat(all_res)
    res_df.to_csv("./results_grid_search.csv")


def main(gs):
    csv_save_dir = "/scratch/usr/nimanwai/experiments/unetr/grid_search/bt-ct-ds/"
    os.makedirs(csv_save_dir, exist_ok=True)

    if gs:
        # set the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # let's get the model
        model = common.get_unetr_model(
            model_name="vit_b", source_choice="torch-em", patch_shape=(512, 512),
            sam_initialization=True, output_channels=3
        )

        # trained model checkpoint
        model_ckpt = "/scratch/usr/nimanwai/experiments/unetr/try/"  # save_root
        model_ckpt += "vit_b/distances/dicebaseddistloss/torch-em-sam/"  # dir structure
        model_ckpt += "checkpoints/livecell-all/best.pt"  # saved checkpoint location

        # let's initialize the model with the trained weights
        model.load_state_dict(torch.load(model_ckpt, map_location=torch.device('cpu'))["model_state"])
        model.to(device)
        model.eval()

        input_dir = "/scratch/usr/nimanwai/data/livecell"
        do_grid_search(model, device, input_dir=input_dir, csv_save_dir=csv_save_dir)

    else:
        check_gs_results(csv_save_dir)


if __name__ == "__main__":
    main(gs=True)
