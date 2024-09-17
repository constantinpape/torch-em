import os
from tqdm import tqdm
from pathlib import Path

import h5py
import numpy as np

import torch

import torch_em
from torch_em.loss import DiceLoss
from torch_em.transform.raw import normalize_percentile
from torch_em.util.prediction import predict_with_halo
from torch_em.util.segmentation import watershed_from_components

from elf.evaluation import mean_segmentation_accuracy

from common import (
    get_dataloaders, get_model, get_experiment_name, get_test_images, dice_score, _load_image, SAVE_DIR
)

# Results:
# GONUCLEAR
# InstanceNormTrackStats: 0.4309 (instance segmentation), 0.8759 (binary segmentation)
# InstanceNorm: 0.3932 (instance segmentation), 0.8981 (binary segmentation)

# PLANTSEG
# TODO

# MITOEM
# TODO

# LIVECELL
# TODO


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

    image_paths, _ = get_test_images(dataset=dataset)

    pred_dir = os.path.join(Path(save_root).parent, "prediction", dataset, norm, task)
    os.makedirs(pred_dir, exist_ok=True)

    _whole_vol_norm = True  # NOTE: performs normalization on either the whole volume or per tile.

    for image_path in tqdm(image_paths, desc="Predicting"):
        pred_path = os.path.join(pred_dir, f"{Path(image_path).stem}.h5")
        if os.path.exists(pred_path):
            continue

        load_kwargs = {}
        if dataset == "gonuclear":
            load_kwargs["key"] = "raw/nuclei"
        elif dataset in ["plantseg", "mitoem"]:
            load_kwargs["key"] = "raw"

        image = _load_image(image_path, **load_kwargs)

        if _whole_vol_norm:
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
            preprocess=None if _whole_vol_norm else normalize_percentile,
        )

        prediction = prediction.squeeze()

        # save outputs
        with h5py.File(pred_path, "a") as f:
            f.create_dataset(
                "segmentation", shape=prediction.shape, data=prediction, compression="gzip", compression_opts=9,
            )


def run_evaluation(norm, dataset, task, save_root):
    visualize = True

    image_paths, gt_paths = get_test_images(dataset=dataset)

    dsc_list, msa_list, sa50_list = [], [], []
    for image_path, gt_path in tqdm(zip(image_paths, gt_paths), desc="Evaluating", total=len(image_paths)):
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

        pred_path = os.path.join(
            Path(save_root).parent, "prediction", dataset, norm, task, f"{Path(image_path).stem}.h5"
        )

        with h5py.File(pred_path, "r") as f:
            if task == "boundaries":
                prediction = f['segmentation'][:]
                if dataset == "plantseg":  # we only have boundary segmentation here.
                    bd = prediction
                    fg = np.ones_like(bd)
                else:
                    fg, bd = prediction

                instances = watershed_from_components(boundaries=bd, foreground=fg)

                msa, sa = mean_segmentation_accuracy(segmentation=instances, groundtruth=gt, return_accuracies=True)
                msa_list.append(msa)
                sa50_list.append(sa[0])

                if visualize:
                    import napari
                    v = napari.Viewer()
                    v.add_image(image, name="Input Image")
                    v.add_image(fg, name="Foreground")
                    v.add_image(bd, name="Boundary")
                    v.add_labels(instances, name="Instances")
                    v.add_labels(gt, name="Ground Truth", visible=False)
                    napari.run()

            else:
                prediction = f["segmentation"][:]
                prediction = (prediction > 0.5)  # threshold the predictions

                gt = (gt > 0)   # binarise the instances

                if visualize:
                    import napari
                    v = napari.Viewer()
                    v.add_image(image, name="Input Image")
                    v.add_labels(prediction, name="Foreground")
                    v.add_labels(gt, name="Ground Truth", visible=False)
                    napari.run()

                score = dice_score(gt=gt, seg=prediction)
                assert score > 0 and score <= 1  # HACK: sanity check
                dsc_list.append(score)

    if task == "binary":
        mean_dice = np.mean(dsc_list)
        print(mean_dice)

    else:
        mean_msa = np.mean(msa_list)
        mean_sa50 = np.mean(sa50_list)
        print(mean_msa, mean_sa50)


def run_analysis_per_dataset(dataset, task, save_root):
    k = 50  # determines the number of images to visualize

    image_paths, gt_paths = get_test_images(dataset=dataset)

    exp1_dir = os.path.join(Path(save_root).parent, "prediction", dataset, "InstanceNorm", task)
    exp2_dir = os.path.join(Path(save_root).parent, "prediction", dataset, "InstanceNormTrackStats", task)

    dice_2d_samples = []
    image_ids = []
    for image_path, gt_path in tqdm(zip(image_paths, gt_paths), desc="Analysing", total=len(image_paths)):
        if dataset == "livecell":
            image = _load_image(image_path)
            gt = _load_image(gt_path)
        elif dataset in ["gonuclear", "plantseg"]:
            image = _load_image(image_path, "raw")
            gt = _load_image(image_path, "label")
        else:  # mitoem
            image = _load_image(image_path, "raw")
            gt = _load_image(image_path, "labels")

        image_id = Path(image_path).stem
        pred_exp1_path = os.path.join(exp1_dir, f"{image_id}.h5")
        pred_exp2_path = os.path.join(exp2_dir, f"{image_id}.h5")

        with h5py.File(pred_exp1_path, "r") as f1:
            if task == "boundaries":
                fg_exp1 = f1["segmentation/foreground"][:]
                bd_exp1 = f1["segmentation/boundary"][:]
            else:
                fg_exp1 = f1["segmentation/foreground"][:]

        with h5py.File(pred_exp2_path, "r") as f2:
            if task == "boundaries":
                fg_exp2 = f2["segmentation/foreground"][:]
                bd_exp2 = f2["segmentation/boundary"][:]
            else:
                fg_exp2 = f2["segmentation/foreground"][:]

        # NOTE: visualize the whole volume as it is
        # import napari
        # v = napari.Viewer()
        # v.add_image(image)
        # v.add_image(fg_exp1, name="olddefault", visible=False)
        # v.add_image(fg_exp2, name="InstanceNorm", visible=False)
        # v.add_labels(gt)
        # napari.run()

        # continue

        if image.ndim == 3:
            # let's check the 10 worst performing slices on "OldDefault" and compare it with "InstanceNorm"
            dice_scores = [
                dice_score(gslice > 0, pslice > 0.5) for pslice, gslice in tqdm(zip(fg_exp1, gt), total=image.shape[0])
            ]
            k_worst = _get_k_worst_indices(dice_scores, k)

            # now, let's visualize the respective slices
            for idx, (islice, p1slice, p2slice, gslice) in enumerate(zip(image, fg_exp1, fg_exp2, gt)):
                if idx not in k_worst:
                    continue

                import napari
                v = napari.Viewer()
                v.add_image(islice)
                v.add_image(gslice, name="GT", visible=False)
                v.add_image(p1slice, name="OldDefault", visible=False)
                v.add_image(p2slice, name="InstanceNorm", visible=False)
                napari.run()

        else:
            # store the dice score pair per experiment and visualize later
            dice_2d_samples.append(dice_score(gt > 0, fg_exp1 > 0.5))
            image_ids.append(image_id)

            # NOTE: visualizing each 2d image
            # import napari
            # v = napari.Viewer()
            # v.add_image(image)
            # v.add_labels(gt)
            # v.add_image(fg_exp1)
            # napari.run()

    if len(dice_2d_samples) > 0:
        k_worst = _get_k_worst_indices(dice_2d_samples, k)
        for idx in k_worst:
            image_path = image_paths[idx]
            gt_path = gt_paths[idx]

            image_id = Path(image_path).stem

            assert image_id == image_ids[idx]

            pred_exp1_path = os.path.join(exp1_dir, f"{image_id}.h5")
            pred_exp2_path = os.path.join(exp2_dir, f"{image_id}.h5")

            with h5py.File(pred_exp1_path, "r") as f3:
                if task == "boundaries":
                    fg_exp1 = f3["segmentation/foreground"][:]
                    bd_exp1 = f3["segmentation/boundary"][:]
                else:
                    fg_exp1 = f3["segmentation/foreground"][:]

            with h5py.File(pred_exp2_path, "r") as f4:
                if task == "boundaries":
                    fg_exp2 = f4["segmentation/foreground"][:]
                    bd_exp2 = f4["segmentation/boundary"][:]
                else:
                    fg_exp2 = f4["segmentation/foreground"][:]

            import napari
            v = napari.Viewer()
            v.add_image(_load_image(image_path))
            v.add_labels(_load_image(gt_path), name="GT")
            v.add_image(fg_exp1, name="OldDefault", visible=False)
            v.add_image(fg_exp2, name="InstanceNorm", visible=False)
            napari.run()


def _get_k_worst_indices(input_list, k):
    _array = np.array(input_list)
    non_zero_mask = (_array != 0)
    non_zero_indices = np.where(non_zero_mask)[0]
    non_zero_values = _array[non_zero_mask]
    bottom_k_non_zero_indices = non_zero_indices[np.argsort(non_zero_values)[:k]]
    return bottom_k_non_zero_indices


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
