import argparse
import os

import imageio
import h5py
import numpy as np

from bioimageio.core import load_resource_description
from bioimageio.core.prediction import predict_with_padding
from bioimageio.core.prediction_pipeline import create_prediction_pipeline
from elf.segmentation.mutex_watershed import mutex_watershed
from elf.segmentation.watershed import apply_size_filter


def predict(path, model, output_path, device):
    image = imageio.imread(path)
    if image.ndim == 2:
        input_ = image[None, None]
    elif image.ndim == 3:
        input_ = image[None]
    assert input_.ndim == 4
    assert input_.shape[:2] == (1, 1)

    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_resource_description(model)
    pp = create_prediction_pipeline(bioimageio_model=model, devices=[device])

    pred = predict_with_padding(pp, [input_], padding={"x": 16, "y": 16})[0]
    assert pred.shape[0] == 1
    fg, affs = np.array(pred[0, 0]), np.array(pred[0, 1:])

    with h5py.File(output_path, "a") as f:
        ds = f.require_dataset("foreground", shape=fg.shape, compression="gzip", dtype=fg.dtype)
        ds[:] = fg
        ds = f.require_dataset("affinities", shape=affs.shape, compression="gzip", dtype=affs.dtype)
        ds[:] = affs

    offsets = model.config["mws"]["offsets"]
    assert len(offsets) == affs.shape[0], f"{len(offsets)}, {affs.shape[0]}"
    return image, fg, affs, offsets


def postprocess(seg, foreground, affinities, offsets, min_size):
    hmap = np.max(affinities[:foreground.ndim], axis=0)
    assert hmap.shape == foreground.shape
    seg, _ = apply_size_filter(seg + 1, hmap, min_size, exclude=[1])
    seg[seg == 1] = 0

    # TODO implement more postprocessing:
    # - merge noise (that only has very weak boundary predictions) into the background

    return seg


def segment(foreground, affinities, offsets, output_path, min_size):
    mask = foreground >= 0.5
    strides = [4] * foreground.ndim
    seg = mutex_watershed(affinities, offsets=offsets, mask=mask, strides=strides, randomize_strides=True)
    seg = postprocess(seg.astype("uint32"), foreground, affinities, offsets, min_size)
    with h5py.File(output_path, "a") as f:
        ds = f.require_dataset("segmentation", shape=seg.shape, compression="gzip", dtype=seg.dtype)
        ds[:] = seg
    return seg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", required=True, help="the filepath for the input image")
    parser.add_argument("--model", "-m", required=True, help="the affinity model for segmentation")
    parser.add_argument("-o", "--output_path", help="the filepath for the result", default=None)
    parser.add_argument("--min_size", help="minimal segment size", default=250)
    parser.add_argument("-d", "--device", help="the device to use for prediction", default=None)
    parser.add_argument("-v", "--view", help="show the prediction results (needs napari)", default=0)
    args = parser.parse_args()
    if args.output_path is None:
        output_path = os.path.splitext(args.path)[0] + ".h5"
    else:
        output_path = args.output_path
    image, foreground, affinities, offsets = predict(args.path, args.model, output_path, args.device)
    seg = segment(foreground, affinities, offsets, output_path, args.min_size)
    if args.view:
        import napari
        v = napari.Viewer()
        v.add_image(image)
        v.add_image(foreground, visible=False)
        v.add_image(affinities, visible=False)
        v.add_labels(seg)
        napari.run()


if __name__ == "__main__":
    main()
