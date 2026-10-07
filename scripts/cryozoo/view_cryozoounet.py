import os
import argparse

import mrcfile
import numpy as np

import napari

import torch

from torch_em.util import load_model

from train_cryozoounet import cryosiam_raw_transform


def predict_crop(checkpoint_dir, raw):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(checkpoint_dir, name="latest", device=device).eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(cryosiam_raw_transform(raw))[None, None].to(device))
    return pred[0, 0].cpu().numpy()


def view(args):
    bb = tuple(slice(start, start + size) for start, size in zip(args.start, args.crop_shape))
    with mrcfile.mmap(args.raw_path, "r", permissive=True) as f:
        raw = np.array(f.data[bb])
    with mrcfile.mmap(args.label_path, "r", permissive=True) as f:
        labels = np.array(f.data[bb]).astype("uint8")

    pred = predict_crop(os.path.join(args.save_root, "checkpoints", args.name), raw)

    v = napari.Viewer()
    v.add_image(cryosiam_raw_transform(raw), name="raw (inverted, scaled)")
    v.add_image(pred, name="prediction", colormap="magma", blending="additive", contrast_limits=(0, 1))
    v.add_labels(labels, name="actin labels").contour = 1
    napari.run()


def main():
    parser = argparse.ArgumentParser(description="Show the prediction of a finetuned CryoZooUNet in napari.")
    parser.add_argument("--raw_path", required=True, help="The tomogram in mrc format.")
    parser.add_argument("--label_path", required=True, help="The binary labels in mrc format.")
    parser.add_argument("--save_root", required=True, help="The folder with the checkpoints.")
    parser.add_argument("--name", default="cryozoounet", help="The name of the checkpoint.")
    parser.add_argument("--start", type=int, nargs=3, default=[400, 48, 48], help="The start of the crop.")
    parser.add_argument("--crop_shape", type=int, nargs=3, default=[64, 256, 256], help="It must be divisible by 8.")
    args = parser.parse_args()
    view(args)


if __name__ == "__main__":
    main()
