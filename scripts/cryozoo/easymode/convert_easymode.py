"""Convert an easymode Keras model to PyTorch via ONNX.

The script needs a specific environment, create one using the yaml in this directory:

    micromamba env create -f environment.yaml
    micromamba activate easymode-convert

Usage:
    python convert_easymode.py --feature actin --output_dir converted_models

Loading a converted model with 'torch.load(path, weights_only=False)' needs onnx2torch.
"""
import os
import glob
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
    with tempfile.TemporaryDirectory() as tmp_dir:
        with tarfile.open(path, "r") as archive:
            archive.extractall(path=tmp_dir)
        model = tf.keras.models.load_model(glob.glob(os.path.join(tmp_dir, "*_weights.h5"))[0], compile=False)
        with open(glob.glob(os.path.join(tmp_dir, "*_metadata.json"))[0]) as f:
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


def to_onnx(model, input_shape, opset):
    """Convert the Keras model to ONNX for a fixed input shape.

    Without a fixed input shape, tf2onnx writes 'SAME_UPPER' padding, which onnx2torch does not support.
    """
    spec = tf.TensorSpec(input_shape, tf.float32, name="input")
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=[spec], opset=opset)
    return onnx_model


def check(reference, torch_model, input_shape, axes, atol=1e-4):
    """Compare the predictions of the Keras reference model and torch for a random input."""
    to_keras, to_torch = axes
    x = np.random.default_rng(0).standard_normal(input_shape, dtype="float32")
    x_keras = np.transpose(x, to_keras).astype(reference.dtype)
    y_keras = np.transpose(reference(x_keras, training=False).numpy(), to_torch)

    with torch.no_grad():
        y_torch = torch_model(torch.from_numpy(x)).numpy()

    print(f"Input shape {x.shape}, output shape {y_keras.shape}.")
    diff = np.abs(y_torch - y_keras).max()
    print(f"Max abs difference to Keras ({reference.dtype}): {diff:.2e}")
    assert np.allclose(y_torch, y_keras, atol=atol), "The torch predictions differ from Keras."
    return diff


def convert_easymode(args):
    weights_path, metadata = get_model(args.feature, variant=args.variant)
    if weights_path is None:
        raise RuntimeError(f"Could not get the easymode model for '{args.feature}'.")
    axes = AXES[metadata["tag"]]

    if weights_path.endswith(".scnm"):
        model, metadata = load_scnm(weights_path)
        # Conversion requires fixed input shape; open dims set to `box_size` used in training, batch size set to 1.
        keras_shape = (1,) + tuple(metadata["box_size"] if sh is None else sh for sh in model.input_shape[1:])
        reference = model
    else:
        # The 'sv3-3d' weights load into the easymode model which is built from `arch` in the .json metadata.
        model = load_model_weights(weights_path)
        keras_shape = (1,) + tuple(get_arch(metadata["arch"])["input_shape"])
        # float32 GroupNorm sums over large volumes are inaccurate on CPU, so the reference runs in float64.
        tf.keras.backend.set_floatx("float64")
        reference = load_model_weights(weights_path)
        tf.keras.backend.set_floatx("float32")

    wrapped = swap_axes(model, keras_shape, axes)
    input_shape = wrapped.input_shape
    onnx_model = to_onnx(wrapped, input_shape, args.opset)
    torch_model = convert(onnx_model).eval()
    diff = check(reference, torch_model, input_shape, axes)

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
    parser.add_argument("--opset", type=int, default=17, help="The ONNX opset version.")
    args = parser.parse_args()
    convert_easymode(args)


if __name__ == "__main__":
    main()
