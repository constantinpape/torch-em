import os
from functools import partial
from glob import glob

import imageio
import h5py
import numpy as np
import torch

from elf.evaluation import mean_average_precision
from torch_em.util import get_trainer
from torch_em.transform.raw import standardize
from tqdm import tqdm

import segmentation_impl

INPUT_FOLDER = "/g/kreshuk/pape/Work/data/data_science_bowl/dsb2018/test"
PRED_FOLDER = "/scratch/pape/dsb/spoco/test"

OFFSETS = [
    [-1, 0], [0, -1],
    [-3, 0], [0, -3],
    [-9, 0], [0, -9],
    [-27, 0], [0, -27],
]


def predict(model, image, device):
    shape = image.shape
    image = standardize(image)
    if any(sh % 16 != 0 for sh in image.shape):
        pad_width = [[0, 0 if sh % 16 == 0 else (16 - (sh % 16))] for sh in image.shape]
        crop = tuple(slice(0, sh) for sh in image.shape)
        image = np.pad(image, pad_width, mode="symmetric")
    else:
        crop = None
    input_ = torch.from_numpy(image[None, None]).to(device)
    pred = model(input_)
    pred = pred[0].detach().cpu().numpy()
    if crop is not None:
        pred = pred[(slice(None),) + crop]
    assert pred.shape[1:] == shape, f"{pred.shape}, {shape}"
    return pred


def run_prediction(checkpoint, seg_name, seg_function):
    os.makedirs(PRED_FOLDER, exist_ok=True)
    inputs = glob(os.path.join(INPUT_FOLDER, "images", "*.tif"))

    pred_key = f"embeddings/{checkpoint}"
    seg_key = f"segmentations/{seg_name}/{checkpoint}"

    with torch.no_grad():
        trainer = get_trainer(os.path.join("./checkpoints", checkpoint))
        model = trainer.model
        device = trainer.device
        model.eval()
        for path in tqdm(inputs, desc=f"Run prediction for {checkpoint}"):
            fname = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(PRED_FOLDER, f"{fname}.h5")
            with h5py.File(out_path, "a") as f:
                if seg_key in f:
                    continue

                if pred_key in f:
                    pred = f[pred_key][:]
                else:
                    image = np.asarray(imageio.imread(path))
                    pred = predict(model, image, device)
                    f.create_dataset(pred_key, data=pred, compression="gzip")

                seg = seg_function(pred)
                f.create_dataset(seg_key, data=seg, compression="gzip")


def evaluate_seg(mask, seg):
    assert mask.shape == seg.shape, f"{mask.shape}, {seg.shape}"
    m_ap, aps = mean_average_precision(seg.astype("uint64"), mask.astype("uint64"), return_aps=True)
    # aps:
    # 0   1    2   3    4   5
    # 0.5 0.55 0.6 0.65 0.7 0.75
    return m_ap, aps[0], aps[5]


def run_evaluation(checkpoint, seg_name):
    masks = glob(os.path.join(INPUT_FOLDER, "images", "*.tif"))
    results = []
    for path in tqdm(masks, desc=f"Run evaluation for {checkpoint}"):
        fname = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(PRED_FOLDER, f"{fname}.h5")
        mask = imageio.imread(path)
        with h5py.File(out_path, "r") as f:
            seg = f[f"segmentations/{seg_name}/{checkpoint}"][:]
        res = evaluate_seg(mask, seg)
        results.append(res)

    metric_names = ["mAP", "IOU50", "IOU75"]
    result = np.array(results).mean(axis=0)
    assert len(metric_names) == len(result)

    return {name: res for name, res in zip(metric_names, result)}


def main():
    eval_results = {}
    checkpoint_names = os.listdir("./checkpoints")

    seg_name = "mws"
    if seg_name == "hdbscan":
        seg_function = segmentation_impl.segment_hdbscan
    elif seg_name == "mws":
        seg_function = partial(
            segmentation_impl.segment_embeddings_mws,
            offsets=OFFSETS,
            distance_type="l2"
        )
    else:
        raise ValueError

    for name in checkpoint_names:
        if name in eval_results:
            continue
        run_prediction(name, seg_name, seg_function)
        eval_results[name] = run_evaluation(name, seg_name)

    for k, v in eval_results.items():
        print(k, ":", v)


if __name__ == "__main__":
    main()
