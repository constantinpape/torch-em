import argparse
import os
from glob import glob
from pathlib import Path

import imageio
import h5py
import pandas as pd

from bioimageio.core import load_resource_description
from bioimageio.core.prediction import predict_with_padding
from bioimageio.core.prediction_pipeline import create_prediction_pipeline
from elf.evaluation import mean_average_precision
from torch_em.util.segmentation import (connected_components_with_boundaries,
                                        mutex_watershed, size_filter)
from tqdm import tqdm
from xarray import DataArray

try:
    import napari
except ImportError:
    napari = None


def segment(prediction_pipeline, path, out_path, view, offsets=None, strides=None, min_seg_size=50):
    image = imageio.imread(path)
    assert image.ndim == 2
    input_ = DataArray(image[None, None], dims=prediction_pipeline.input_specs[0].axes)
    padding = {"x": 16, "y": 16}
    prediction = predict_with_padding(prediction_pipeline, input_, padding)[0][0]
    foreground, prediction = prediction[0], prediction[1:]

    if offsets is None:
        assert prediction.shape[0] == 1, f"{prediction.shape}"
        prediction = prediction[0]
        assert foreground.shape == prediction.shape
        seg = connected_components_with_boundaries(foreground, prediction)
    else:
        assert len(offsets) == prediction.shape[0]
        mask = foreground > 0.5
        seg = mutex_watershed(prediction, offsets, mask=mask, strides=strides)
    seg = size_filter(seg, min_seg_size)

    if out_path is not None:
        with h5py.File(out_path, "w") as f:
            f.create_dataset("prediction", data=prediction, compression="gzip")
            f.create_dataset("foreground", data=foreground, compression="gzip")
            f.create_dataset("segmentation", data=seg, compression="gzip")

    if view:
        assert napari is not None
        v = napari.Viewer()
        v.add_image(image)
        v.add_image(foreground)
        v.add_image(prediction)
        v.add_labels(seg)
        napari.run()

    return seg


def validate(seg, gt_path):
    gt = imageio.imread(gt_path)
    assert gt.shape == seg.shape
    map_, scores = mean_average_precision(seg, gt, return_aps=True)
    # map, iou50, iou75, iou90
    return [map_, scores[0], scores[5], scores[-1]]


def run_prediction(model_path, input_files, target_files, output_folder, view, min_seg_size, device):
    model = load_resource_description(model_path)
    offsets, strides = None, None
    if "mws" in model.config:
        offsets = model.config["mws"]["offsets"]
        strides = [4, 4]

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
    validation_results = []

    devices = None if device is None else [device]
    with create_prediction_pipeline(bioimageio_model=model, devices=devices) as pp:
        for in_path, target_path in tqdm(zip(input_files, target_files), total=len(input_files)):
            fname = str(Path(in_path).stem)
            out_path = None if output_folder is None else os.path.join(output_folder, f"{fname}.h5")
            seg = segment(pp, in_path, out_path, view,
                          offsets=offsets, strides=strides, min_seg_size=min_seg_size)
            if target_path:
                val = validate(seg, target_path)
                validation_results.append([fname] + val)

    if validation_results:
        cols = ["name", "mAP", "IoU50", "IoU75", "IoU90"]
        validation_results = pd.DataFrame(validation_results, columns=cols)
        print("Validation results averaged over all", len(input_files), "images:")
        print(validation_results[cols[1:]].mean(axis=0))
        return validation_results


def _load_data(input_folder, ext):
    input_data = glob(os.path.join(input_folder, "images", f"*.{ext}"))
    input_data.sort()
    if os.path.exists(os.path.join(input_folder, "masks")):
        input_target = glob(os.path.join(input_folder, "masks", f"*.{ext}"))
        input_target.sort()
    else:
        input_target = [None] * len(input_data)
    assert len(input_data) == len(input_target)
    return input_data, input_target


def main():
    parser = argparse.ArgumentParser(
        "Run prediction and segmentation with a bioimagie.io model and save or validate the results."
        "If 'output_folder' is passed, the results will be saved as hdf5 files with keys:"
        "prediction: the affinity or boundary predictions"
        "foreground: the foreground predictions"
        "segmentation: the nucleus instance segmentation"
    )
    parser.add_argument("-m", "--model", required=True, help="Path to the bioimage.io model.")
    parser.add_argument("-i", "--input_folder", required=True,
                        help="The root input folder with subfolders 'images' and (optionally) 'masks'")
    parser.add_argument("--ext", default="tif", help="The file extension of the input files.")
    parser.add_argument("-o", "--output_folder", default=None, help="Where to save the results.")
    parser.add_argument("-v", "--view", default=0,
                        help="Whether to show segmentation results (needs napari).", type=int)
    parser.add_argument("--min_seg_size", default=25, type=int)
    parser.add_argument("--device", default=None, help="The device used for inference.")
    parser.add_argument("--save_path", "-s", default=None, help="Where to save a csv with the validation results.")
    args = parser.parse_args()

    input_files, target_files = _load_data(args.input_folder, args.ext)
    res = run_prediction(args.model, input_files, target_files, args.output_folder,
                         view=bool(args.view), min_seg_size=args.min_seg_size, device=args.device)
    if args.save_path is not None:
        assert res is not None
        res.to_csv(args.save_path, index=False)


if __name__ == "__main__":
    main()
