import os
from tqdm import tqdm
from pathlib import Path

import h5py
import numpy as np
from skimage.segmentation import find_boundaries

import torch

import torch_em
from torch_em.loss import DiceLoss
from torch_em.transform.raw import normalize_percentile
from torch_em.util.prediction import predict_with_halo

from elf.evaluation import mean_segmentation_accuracy

from common import (
    get_dataloaders, get_model, get_experiment_name, get_test_images, dice_score, _load_image, SAVE_DIR
)

# Results:
# GONUCLEAR
# InstanceNorm:           0.2657 (instance segmentation), 0.8981 (foreground segmentation)
# InstanceNormTrackStats: 0.4238 (instance segmentation), 0.8759 (foreground segmentation)

# PLANTSEG
# InstanceNorm:           0.5946 (boundary segmentation)
# InstanceNormTrackStats: 0.5538 (boundary segmentation)

# MITOEM
# InstanceNorm:           ... (instance segmentation), 0.9197 (foreground segmentation)
# InstanceNormTrackStats: ... (instance segmentation), 0.9223 (foreground segmentation)

# LIVECELL
# InstanceNorm:           ... (instance segmentation), ... (foreground segmentation)
# InstanceNormTrackStats: ... (instance segmentation), ... (foreground segmentation)


def run_training(name, model, dataset, task, save_root, device):
    n_iterations = int(2.5e4)
    train_loader, val_loader = get_dataloaders(dataset=dataset, task=task)

    trainer = torch_em.default_segmentation_trainer(
        name=name,
        save_root=save_root,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=DiceLoss(),
        metric=DiceLoss(),
        device=device,
        learning_rate=5e-4,
        mixed_precision=True,
        compile_model=False,
        log_image_interval=100,
    )
    trainer.fit(iterations=n_iterations)


def run_inference(name, model, norm, dataset, task, save_root, device):
    checkpoint = os.path.join(save_root, "checkpoints", name, "best.pt")
    assert os.path.exists(checkpoint), checkpoint

    model.load_state_dict(torch.load(checkpoint, map_location=torch.device("cpu"))["model_state"])
    model.to(device)
    model.eval()

    image_paths, gt_paths = get_test_images(dataset=dataset)

    pred_dir = os.path.join(Path(save_root).parent, "prediction", dataset, norm, task)
    os.makedirs(pred_dir, exist_ok=True)

    for image_path, gt_path in tqdm(zip(image_paths, gt_paths), desc="Predicting", total=len(image_paths)):
        pred_path = os.path.join(pred_dir, f"{Path(image_path).stem}.h5")
        if os.path.exists(pred_path):
            continue

        image, _ = _get_per_dataset_inputs(dataset, image_path, gt_path)
        image = normalize_percentile(image)

        if dataset == "livecell":
            tile, halo = (384, 384), (64, 64)
        else:
            tile, halo = (16, 384, 384), (8, 64, 64)

        prediction = predict_with_halo(
            input_=image,
            model=model,
            gpu_ids=[device],
            block_shape=tile,
            halo=halo,
            preprocess=None,
        )
        prediction = prediction.squeeze()

        # save outputs
        dname = "segmentation/prediction" if task == "boundaries" else "segmentation/foreground"
        with h5py.File(pred_path, "a") as f:
            f.create_dataset(dname, shape=prediction.shape, data=prediction, compression="gzip")


def _get_per_dataset_inputs(dataset, image_path, gt_path):
    if dataset == "livecell":
        image, gt = _load_image(image_path), _load_image(gt_path)
    elif dataset == "gonuclear":
        image, gt = _load_image(image_path, "raw/nuclei"), _load_image(image_path, "labels/nuclei")
    elif dataset == "plantseg":
        image, gt = _load_image(image_path, "raw"), _load_image(image_path, "label")
    elif dataset == "mitoem":
        image, gt = _load_image(image_path, "raw"), _load_image(image_path, "labels")
    else:
        raise ValueError

    return image, gt


def run_evaluation(norm, dataset, task, save_root):
    image_paths, gt_paths = get_test_images(dataset=dataset)

    dsc_list, msa_list, sa50_list = [], [], []
    for image_path, gt_path in tqdm(zip(image_paths, gt_paths), desc="Evaluating", total=len(image_paths)):
        _, gt = _get_per_dataset_inputs(dataset, image_path, gt_path)

        pred_path = os.path.join(
            Path(save_root).parent, "prediction", dataset, norm, task, f"{Path(image_path).stem}.h5"
        )
        with h5py.File(pred_path, "r+") as f:
            if task == "boundaries" and dataset != "plantseg":
                prediction = f['segmentation/prediction'][:]
                fg, bd = prediction

                if dataset == "livecell":
                    tile, halo = (384, 384), (64, 64)
                else:
                    tile, halo = (16, 384, 384), (8, 64, 64)

                # NOTE: performing instance segmentation over the entire volume is very slow!
                # instances = watershed_from_components(boundaries=bd, foreground=fg)

                from elf.parallel import seeded_watershed
                from skimage.measure import label as connected_components
                seeds = connected_components((fg - bd) > 0.5)
                mask = fg > 0.5
                instances = seeded_watershed(
                    hmap=bd, seeds=seeds, out=np.zeros_like(gt), block_shape=tile, halo=halo, mask=mask, verbose=True,
                )

                msa, sa = mean_segmentation_accuracy(segmentation=instances, groundtruth=gt, return_accuracies=True)
                msa_list.append(msa)
                sa50_list.append(sa[0])

                # f.create_dataset("segmentation/foreground", shape=fg.shape, data=fg, compression="gzip")
                # f.create_dataset("segmentation/boundary", shape=bd.shape, data=bd, compression="gzip")
                # f.create_dataset("segmentation/instances", shape=instances.shape, data=instances, compression="gzip")

            else:
                if dataset == "plantseg":
                    prediction = f["segmentation/prediction"][:]
                    gt = find_boundaries(gt)
                else:
                    prediction = f["segmentation/foreground"][:]
                    gt = (gt > 0)   # binarise the instances

                score = dice_score(gt=gt, seg=prediction > 0.5)
                assert score > 0 and score <= 1  # HACK: sanity check
                dsc_list.append(score)

    if len(dsc_list) > 0:
        mean_dice = np.mean(dsc_list)
        print(f"Mean dice score: {mean_dice}")
    else:
        mean_msa = np.mean(msa_list)
        mean_sa50 = np.mean(sa50_list)
        print(f"Mean mSA: {mean_msa}, mean SA50: {mean_sa50}")


def run_analysis_per_dataset(dataset, task, save_root):
    exp1_dir = os.path.join(Path(save_root).parent, "prediction", dataset, "InstanceNorm", task)
    exp2_dir = os.path.join(Path(save_root).parent, "prediction", dataset, "InstanceNormTrackStats", task)

    image_paths, gt_paths = get_test_images(dataset=dataset)
    for image_path, gt_path in tqdm(zip(image_paths, gt_paths), desc="Analysing", total=len(image_paths)):
        image, gt = _get_per_dataset_inputs(dataset, image_path, gt_path)

        if dataset == "plantseg":
            gt = find_boundaries(gt)

        image_id = Path(image_path).stem
        pred_exp1_path = os.path.join(exp1_dir, f"{image_id}.h5")
        pred_exp2_path = os.path.join(exp2_dir, f"{image_id}.h5")

        with h5py.File(pred_exp1_path, "r") as f1:
            fg_exp1 = f1["segmentation/foreground"][:]
            if task == "boundaries":
                bd_exp1 = f1["segmentation/boundary"][:]
                instances_exp1 = f1["segmentation/instances"][:]

        with h5py.File(pred_exp2_path, "r") as f2:
            fg_exp2 = f2["segmentation/foreground"][:]
            if task == "boundaries":
                bd_exp2 = f2["segmentation/boundary"][:]
                instances_exp2 = f2["segmentation/instances"][:]

        import napari
        v = napari.Viewer()
        v.add_image(image, name="Image")
        v.add_labels(gt, name="Ground Truth Labels", visible=False)
        v.add_image(fg_exp1, name="Foreground (InstanceNorm)", visible=False)
        v.add_image(fg_exp2, name="Foreground (InstanceNormTrackStats)", visible=False)
        if task == "boundaries":
            v.add_image(bd_exp1, name="Boundary (InstanceNorm)", visible=False)
            v.add_image(bd_exp2, name="Boundary (InstanceNormTrackStats)", visible=False)
            v.add_image(instances_exp1, name="Instance Segmentation (InstanceNorm)")
            v.add_image(instances_exp2, name="Instance Segmentation (InstanceNormTrackStats)")
        napari.run()


def main(args):
    phase = args.phase
    dataset = args.dataset
    task = args.task
    norm = args.norm

    if dataset == "plantseg" and task == "binary":
        raise ValueError("The experiment for binary segmentation of 'PlantSeg (Root)' is not implemented.")

    assert task in ["binary", "boundaries"]
    save_root = os.path.join(SAVE_DIR, "models")

    if phase == "evaluate":
        assert norm in ["InstanceNormTrackStats", "InstanceNorm"]
        run_evaluation(norm=norm, dataset=dataset, task=task, save_root=save_root)

    elif phase == "analysis":
        run_analysis_per_dataset(dataset=dataset, task=task, save_root=save_root)

    else:
        assert norm in ["InstanceNormTrackStats", "InstanceNorm"]
        model = get_model(dataset=dataset, task=task, norm=norm)
        name = get_experiment_name(dataset=dataset, task=task, norm=norm, model_choice="unet")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if phase == "train":
            run_training(name=name, model=model, dataset=dataset, task=task, save_root=save_root, device=device)
        elif phase == "predict":
            run_inference(
                name=name, model=model, norm=norm, dataset=dataset, task=task, save_root=save_root, device=device
            )
        else:
            raise ValueError


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, type=str, help="The choice of dataset.")
    parser.add_argument("-p", "--phase", required=True, type=str, help="The mode for running the scripts.")
    parser.add_argument("-t", "--task", required=True, type=str, help="The type of task for segmentation.")
    parser.add_argument("-n", "--norm", type=str, help="The choice of layer normalization.")
    args = parser.parse_args()
    main(args)