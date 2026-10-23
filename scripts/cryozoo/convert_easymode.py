import os
import glob
import json
import tarfile
import argparse
import tempfile

import numpy as np

import onnx
import onnxruntime
import tensorflow as tf
import tf2onnx
import torch
from onnx2torch import convert

from easymode.core.distribution import get_model


def load_scnm(path):
    """Load the Keras model and the metadata from an Ais .scnm archive."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        with tarfile.open(path, "r") as archive:
            archive.extractall(path=tmp_dir)
        model = tf.keras.models.load_model(glob.glob(os.path.join(tmp_dir, "*_weights.h5"))[0], compile=False)
        with open(glob.glob(os.path.join(tmp_dir, "*_metadata.json"))[0]) as f:
            metadata = json.load(f)
    return model, metadata


# The axis permutations from torch (B, C, [Z,] Y, X) to Keras (B, Y, X, [Z,] C) and back, by the number of dimensions.
# Ais feeds the Keras models with (Y, X, Z) volumes, see 'SEModel.apply_to_slice'.
AXES = {4: ((0, 2, 3, 1), (0, 3, 1, 2)), 5: ((0, 3, 4, 2, 1), (0, 4, 3, 1, 2))}


def swap_axes(model):
    """Wrap the Keras model, so that it takes and returns the axis order of torch-em."""
    to_keras, to_torch = AXES[len(model.input_shape)]
    torch_shape = tuple(model.input_shape[i] for i in to_torch)
    inputs = tf.keras.Input(shape=torch_shape[1:], batch_size=torch_shape[0])
    x = tf.keras.layers.Permute(to_keras[1:])(inputs)
    outputs = tf.keras.layers.Permute(to_torch[1:])(model(x))
    return tf.keras.Model(inputs, outputs)


def to_onnx(model, input_shape, opset):
    """Convert the Keras model to ONNX for a static input shape.

    With dynamic spatial dimensions, tf2onnx writes 'SAME_UPPER' padding, which onnx2torch does not support.
    """
    spec = tf.TensorSpec(input_shape, tf.float32, name="input")
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=[spec], opset=opset)
    return onnx_model


def check(model, onnx_model, torch_model, input_shape, atol=1e-4):
    """Compare the predictions of the original Keras model, ONNX runtime and torch for a random input."""
    to_keras, to_torch = AXES[len(input_shape)]
    x = np.random.default_rng(0).standard_normal(input_shape, dtype="float32")
    y_keras = np.transpose(model(np.transpose(x, to_keras), training=False).numpy(), to_torch)

    session = onnxruntime.InferenceSession(onnx_model.SerializeToString(), providers=["CPUExecutionProvider"])
    y_onnx = session.run(None, {session.get_inputs()[0].name: x})[0]

    with torch.no_grad():
        y_torch = torch_model(torch.from_numpy(x)).numpy()

    print(f"Input shape {x.shape}, output shape {y_keras.shape}.")
    diff_onnx, diff_torch = np.abs(y_onnx - y_keras).max(), np.abs(y_torch - y_keras).max()
    print(f"Max abs difference to Keras: ONNX {diff_onnx:.2e}, torch {diff_torch:.2e}")
    assert np.allclose(y_onnx, y_keras, atol=atol), "The ONNX predictions differ from Keras."
    assert np.allclose(y_torch, y_keras, atol=atol), "The torch predictions differ from Keras."


def convert_easymode(args):
    weights_path, _ = get_model(args.feature, variant=args.variant)
    if weights_path is None:
        raise RuntimeError(f"Could not get the easymode model for '{args.feature}'.")
    if not weights_path.endswith(".scnm"):
        raise ValueError(f"Only .scnm models are supported, got {weights_path}.")

    model, metadata = load_scnm(weights_path)
    wrapped = swap_axes(model)
    # Dimensions that the Keras model leaves open are filled with the training box size.
    input_shape = tuple(metadata["box_size"] if sh is None else sh for sh in wrapped.input_shape)
    onnx_model = to_onnx(wrapped, input_shape, args.opset)
    torch_model = convert(onnx_model).eval()
    check(model, onnx_model, torch_model, input_shape)

    os.makedirs(args.output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(weights_path))[0]
    onnx.save(onnx_model, os.path.join(args.output_dir, f"{name}.onnx"))
    torch.save(torch_model, os.path.join(args.output_dir, f"{name}.pt"))
    with open(os.path.join(args.output_dir, f"{name}.json"), "w") as f:
        json.dump({**metadata, "input_layout": "channels_first", "input_shape": input_shape}, f, indent=2)
    print(f"Saved the converted model to {args.output_dir}.")


def main():
    parser = argparse.ArgumentParser(description="Convert an easymode .scnm model to PyTorch via ONNX.")
    parser.add_argument("--feature", required=True, help="The easymode feature, e.g. 'actin'.")
    parser.add_argument("--variant", help="The model variant. By default, easymode uses the default variant.")
    parser.add_argument("--output_dir", required=True, help="The folder for the converted model.")
    parser.add_argument("--opset", type=int, default=17, help="The ONNX opset version.")
    args = parser.parse_args()
    convert_easymode(args)


if __name__ == "__main__":
    main()
