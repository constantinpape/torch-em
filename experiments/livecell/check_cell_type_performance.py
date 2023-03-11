import os
from glob import glob

import numpy as np
import pandas as pd
try:
    import imageio.v2 as imageio
except ImportError:
    import imageio

from tqdm import tqdm
from xarray import DataArray
from elf.evaluation import dice_score


def run_prediction(input_folder, output_folder):
    import bioimageio.core
    os.makedirs(output_folder, exist_ok=True)

    inputs = glob(os.path.join(input_folder, "*.tif"))
    model = bioimageio.core.load_resource_description("10.5281/zenodo.5869899")

    with bioimageio.core.create_prediction_pipeline(model) as pp:
        for inp in tqdm(inputs):
            fname = os.path.basename(inp)
            out_path = os.path.join(output_folder, fname)
            image = imageio.v2.imread(inp)
            input_ = DataArray(image[None, None], dims=tuple("bcyx"))
            pred = bioimageio.core.predict_with_padding(pp, input_)[0].values.squeeze()
            imageio.volwrite(out_path, pred)


def evaluate(label_folder, output_folder):
    cell_types = ["A172", "BT474", "BV2", "Huh7",
                  "MCF7", "SHSY5Y", "SkBr3", "SKOV3"]
    grid = pd.DataFrame(columns=["Cell_types"] + cell_types)
    row = ["all"]
    for i in cell_types:
        label_files = glob(os.path.join(label_folder, i, "*.tif"))
        this_scores = []
        for label_file in label_files:
            fname = os.path.basename(label_file)
            pred_file = os.path.join(output_folder, fname)
            label = imageio.imread(label_file)
            pred = imageio.volread(pred_file)[0]
            score = dice_score(pred, label != 0, threshold_gt=None, threshold_seg=None)

            this_scores.append(score)
        row.append(np.mean(this_scores))

    grid.loc[len(grid)] = row

    print("Cell type results:")
    print(grid)


def main():
    # input_folder = "/home/pape/Work/data/incu_cyte/livecell/images/livecell_test_images"
    output_folder = "./predictions"
    # run_prediction(input_folder, output_folder)
    label_folder = "/home/pape/Work/data/incu_cyte/livecell/annotations/livecell_test_images"
    evaluate(label_folder, output_folder)


if __name__ == "__main__":
    main()
