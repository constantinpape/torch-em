import os
from tqdm import tqdm
from pathlib import Path

import h5py
import numpy as np

import torch

import torch_em
from torch_em.loss import DiceLoss
from torch_em.util.prediction import predict_with_halo
from torch_em.util.segmentation import watershed_from_components

from elf.evaluation import mean_segmentation_accuracy

from common import (
    get_dataloaders, get_model, get_experiment_name, get_test_images, dice_score, _load_image, SAVE_DIR
)


def run_training(name, model, dataset, task, save_root, device):
    train_loader, val_loader = get_dataloaders(dataset=dataset, task=task)

    loss = DiceLoss()

    trainer = torch_em.default_segmentation_trainer(
        name=name,
        save_root=save_root,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=loss,
        device=device,
        learning_rate=1e-4,
        mixed_precision=True,
        compile_model=False,
        log_image_interval=100,
    )
    trainer.fit(iterations=int(1e5))


def run_inference(name, model, norm, dataset, task, save_root, device):
    checkpoint = os.path.join(save_root, "checkpoints", name, "best.pt")
    assert os.path.exists(checkpoint)

    model.load_state_dict(torch.load(checkpoint, map_location=torch.device("cpu"))["model_state"])
    model.to(device)
    model.eval()

    image_paths, _ = get_test_images(dataset=dataset)

    pred_dir = os.path.join(save_root, "prediction", dataset)
    os.makedirs(pred_dir, exist_ok=True)

    for image_path in tqdm(image_paths, desc="Predicting"):
        if dataset == "livecell":
            image = _load_image(image_path)
        elif dataset in ["mouse_embryo", "plantseg", "mitoem"]:
            image = _load_image(image_path, "raw")

        if dataset == "livecell":
            tile, halo = (512, 512), (64, 64)
        else:
            tile, halo = (64, 256, 256), (16, 64, 64)

        prediction = predict_with_halo(
            input_=image, model=model, gpu_ids=[device], block_shape=tile, halo=halo, disable_tqdm=False,
        )

        prediction = prediction.squeeze()

        # save outputs
        image_id = Path(image_path).stem
        pred_path = os.path.join(pred_dir, f"{image_id}.h5")

        with h5py.File(pred_path, "a") as f:
            if task == "boundaries":
                fg, bd = prediction
                f.create_dataset(f"segmentation/{norm}/{task}/foreground", shape=fg.shape, data=fg)
                f.create_dataset(f"segmentation/{norm}/{task}/boundary", shape=bd.shape, data=bd)
            else:
                outputs = prediction
                f.create_dataset(f"segmentation/{norm}/{task}/foreground", shape=outputs.shape, data=outputs)


def run_evaluation(norm, dataset, task, save_root):
    visualize = False

    image_paths, gt_paths = get_test_images(dataset=dataset)

    pred_dir = os.path.join(save_root, "prediction", dataset)

    dsc_list, msa_list, sa50_list = [], [], []
    for image_path, gt_path in tqdm(zip(image_paths, gt_paths), desc="Evaluating", total=len(image_paths)):
        if dataset == "livecell":
            image = _load_image(image_path)
            gt = _load_image(gt_path)
        elif dataset in ["mouse_embryo", "plantseg"]:
            image = _load_image(image_path, "raw")
            gt = _load_image(image_path, "label")

        image_id = Path(image_path).stem
        pred_path = os.path.join(pred_dir, f"{image_id}.h5")

        with h5py.File(pred_path, "r") as f:
            if task == "boundaries":
                fg = f[f"segmentation/{norm}/{task}/foreground"][:]
                bd = f[f"segmentation/{norm}/{task}/boundary"][:]
                instances = watershed_from_components(boundaries=bd, foreground=fg)

                msa, sa = mean_segmentation_accuracy(segmentation=instances, groundtruth=gt, return_accuracies=True)
                msa_list.append(msa)
                sa50_list.append(sa[0])

                if visualize:
                    import napari
                    v = napari.Viewer()
                    v.add_image(image)
                    v.add_labels(instances)
                    v.add_labels(fg > 0.5)
                    v.add_labels(bd > 0.5)
                    v.add_labels(gt, visible=False)
                    napari.run()

            else:
                prediction = f[f"segmentation/{norm}/{task}/foreground"][:]
                prediction = (prediction > 0.5)  # threshold the predictions

                gt = (gt > 0)   # binarise the instances

                if visualize:
                    import napari
                    v = napari.Viewer()
                    v.add_image(image)
                    v.add_labels(prediction)
                    v.add_labels(gt, visible=False)
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


def main(args):
    phase = args.phase
    dataset = args.dataset
    task = args.task
    norm = args.norm

    assert norm in ["OldDefault", "InstanceNorm"]

    save_root = os.path.join(SAVE_DIR, "models")
    model = get_model(dataset=dataset, task=task, norm=norm)
    name = get_experiment_name(dataset=dataset, task=task, norm=norm, model_choice="unet")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if phase == "train":
        run_training(
            name=name, model=model, dataset=dataset, task=task, save_root=save_root, device=device,
        )

    elif phase == "predict":
        run_inference(
            name=name, model=model, norm=norm, dataset=dataset, task=task, save_root=save_root, device=device,
        )

    elif phase == "evaluate":
        run_evaluation(
            norm=norm, dataaset=dataset, task=task, save_root=save_root,
        )

    else:
        print(f"'{phase}' is not a valid mode. Choose from 'train' / 'predict' / 'evaluate'.")


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset", required=True, type=str, help="The choice of dataset."
    )
    parser.add_argument(
        "-p", "--phase", required=True, type=str, help="The mode for running the scripts."
    )
    parser.add_argument(
        "-t", "--task", required=True, type=str, help="The type of task for segmentation."
    )
    parser.add_argument(
        "-n", "--norm", required=True, type=str, help="The choice of layer normalization."
    )
    args = parser.parse_args()
    main(args)
