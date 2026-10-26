"""Convert an easymode Keras model to PyTorch via ONNX.

The script needs a specific environment, create one using the yaml in this directory:

    micromamba env create -f environment.yaml
    micromamba activate easymode-convert

Usage:
    python convert_easymode.py --feature actin --output_dir converted_models

Loading a converted model with 'torch.load(path, weights_only=False)' needs onnx2torch.
"""
import os
import json
import tarfile
import argparse
import tempfile

import numpy as np

import onnx
import tensorflow as tf
import tf2onnx
import torch
from onnx2torch import convert

from easymode.core.distribution import get_model, load_model_weights
from easymode.segmentation.models import get_arch


def load_scnm(path):
    """Load the Keras model and the metadata from an Ais .scnm archive."""
    with tempfile.TemporaryDirectory() as tmp_dir, tarfile.open(path, "r") as archive:
        # Extract only the two files we need, by their base name, so that no member is written outside tmp_dir.
        files = {}
        for suffix in ("_weights.h5", "_metadata.json"):
            members = [m for m in archive.getmembers() if m.isfile() and m.name.endswith(suffix)]
            if len(members) != 1:
                raise RuntimeError(f"Expected one '*{suffix}' file in {path}, found {len(members)}.")
            files[suffix] = os.path.join(tmp_dir, suffix)
            with archive.extractfile(members[0]) as src, open(files[suffix], "wb") as dst:
                dst.write(src.read())
        model = tf.keras.models.load_model(files["_weights.h5"], compile=False)
        with open(files["_metadata.json"]) as f:
            metadata = json.load(f)
    return model, metadata


# The axis permutations from torch (B, C, Z, Y, X) to Keras, by model type.
# Ais 'sv2-3d' model input is (Y, X, Z).
# Ais 'sv2-2.5d' model input is (Y, X, D), with the depth D as channels.
# easymode 'sv3-3d' model input is (Z, Y, X).
AXES = {
    "sv2-3d": ((0, 3, 4, 2, 1), (0, 4, 3, 1, 2)),
    "sv2-2.5d": ((0, 2, 3, 1), (0, 3, 1, 2)),
    "sv3-3d": ((0, 2, 3, 4, 1), (0, 4, 1, 2, 3)),
}


def swap_axes(model, keras_shape, axes):
    """Wrap the Keras model, so that it takes and returns the axis order of torch-em."""
    to_keras, to_torch = axes
    torch_shape = tuple(keras_shape[i] for i in to_torch)
    inputs = tf.keras.Input(shape=torch_shape[1:], batch_size=torch_shape[0])
    x = tf.keras.layers.Permute(to_keras[1:])(inputs)
    outputs = tf.keras.layers.Permute(to_torch[1:])(model(x))
    return tf.keras.Model(inputs, outputs)


# The ONNX operator set version that tf2onnx writes and onnx2torch reads. 17 works for all three model types.
OPSET = 17


def to_onnx(model, input_shape):
    """Convert the Keras model to ONNX for a fixed input shape.

    Without a fixed input shape, tf2onnx writes 'SAME_UPPER' padding, which onnx2torch does not support.
    """
    spec = tf.TensorSpec(input_shape, tf.float32, name="input")
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=[spec], opset=OPSET)
    return onnx_model


def check(reference, torch_model, input_shape, axes, max_atol=None, mean_atol=None):
    """Compare the predictions of the Keras reference model and torch for a random input.

    The check fails if the max or the mean abs difference is above `max_atol` or `mean_atol`, if given.
    """
    to_keras, to_torch = axes
    x = np.random.default_rng(0).standard_normal(input_shape, dtype="float32")
    x_keras = np.transpose(x, to_keras).astype(reference.dtype)
    y_keras = np.transpose(reference(x_keras, training=False).numpy(), to_torch)

    with torch.no_grad():
        y_torch = torch_model(torch.from_numpy(x)).numpy()

    print(f"Input shape {x.shape}, output shape {y_keras.shape}.")
    diff = np.abs(y_torch - y_keras)
    max_diff, mean_diff = diff.max(), diff.mean()
    print(f"Abs difference to Keras ({reference.dtype}): max {max_diff:.2e}, mean {mean_diff:.2e}")
    if max_atol is not None and max_diff > max_atol:
        raise RuntimeError(f"The max abs difference of torch to Keras is {max_diff:.2e}, more than {max_atol}.")
    if mean_atol is not None and mean_diff > mean_atol:
        raise RuntimeError(f"The mean abs difference of torch to Keras is {mean_diff:.2e}, more than {mean_atol}.")
    return max_diff, mean_diff


def convert_easymode(args):
    weights_path, metadata = get_model(args.feature, variant=args.variant)
    if weights_path is None:
        raise RuntimeError(f"Could not get the easymode model for '{args.feature}'.")

    if weights_path.endswith(".scnm"):
        model, scnm_metadata = load_scnm(weights_path)
        # The easymode metadata wins on duplicate keys, it describes the published model.
        metadata = {**scnm_metadata, **metadata}
        # The model type follows from the input rank: (B, Y, X, Z, C) for 'sv2-3d', (B, Y, X, D) for 'sv2-2.5d'.
        axes = AXES["sv2-3d" if len(model.input_shape) == 5 else "sv2-2.5d"]
        # Fixed input shape: open dims set to `box_size` used in training, batch size set to 1.
        keras_shape = (1,) + tuple(scnm_metadata["box_size"] if sh is None else sh for sh in model.input_shape[1:])
        reference = model
        tolerance = {"max_atol": 1e-4}
    else:
        # The 'sv3-3d' weights load into the easymode model which is built from `arch` in the .json metadata.
        # Like 'load_model_weights', fall back to the default arch if `arch` is missing.
        axes = AXES["sv3-3d"]
        model = load_model_weights(weights_path)
        keras_shape = (1,) + tuple(get_arch(metadata.get("arch"))["input_shape"])
        # float32 GroupNorm sums over large volumes are inaccurate on CPU, so the reference runs in float64.
        tf.keras.backend.set_floatx("float64")
        try:
            reference = load_model_weights(weights_path)
        finally:
            tf.keras.backend.set_floatx("float32")
        # The float32 GroupNorm error in torch hits a few voxels by up to ~5e-2, so check the mean instead of the max.
        tolerance = {"mean_atol": 1e-3}

    wrapped = swap_axes(model, keras_shape, axes)
    input_shape = wrapped.input_shape
    onnx_model = to_onnx(wrapped, input_shape)
    torch_model = convert(onnx_model).eval()
    diff = check(reference, torch_model, input_shape, axes, **tolerance)

    os.makedirs(args.output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(weights_path))[0]
    onnx.save(onnx_model, os.path.join(args.output_dir, f"{name}.onnx"))
    torch.save(torch_model, os.path.join(args.output_dir, f"{name}.pt"))
    with open(os.path.join(args.output_dir, f"{name}.json"), "w") as f:
        json.dump({**metadata, "input_layout": "channels_first", "input_shape": input_shape}, f, indent=2)
    print(f"Saved the converted model to {args.output_dir}.")
    return input_shape, diff


def main():
    parser = argparse.ArgumentParser(description="Convert an easymode model to PyTorch via ONNX.")
    parser.add_argument("--feature", required=True, help="The easymode feature, e.g. 'actin'.")
    parser.add_argument("--variant", help="The model variant. By default, easymode uses the default variant.")
    parser.add_argument("--output_dir", required=True, help="The folder for the converted model.")
    args = parser.parse_args()
    convert_easymode(args)


if __name__ == "__main__":
    main()
