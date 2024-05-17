import os
from tqdm import tqdm

import numpy as np
import imageio.v3 as imageio
from skimage.segmentation import find_boundaries

import torch

import torch_em
from torch_em.loss import DiceLoss
from torch_em.util.prediction import predict_with_halo

from common import get_dataloaders, get_model, get_experiment_name, get_test_images, dice_score


SAVE_DIR = "/scratch/projects/nim00007/test/verify_normalization"  # for HLRN
# SAVE_DIR = "/media/anwai/ANWAI/models/torch-em/verify_normalization"


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


def run_inference(name, model, dataset, task, save_root, device):
    checkpoint = os.path.join(save_root, "checkpoints", name, "best.pt")
    assert os.path.exists(checkpoint)

    model.load_state_dict(torch.load(checkpoint, map_location=torch.device("cpu"))["model_state"])
    model.to(device)
    model.eval()

    image_paths, gt_paths = get_test_images(dataset=dataset)

    scores = []
    for image_path, gt_path in tqdm(zip(image_paths, gt_paths), desc="Predicting", total=len(image_paths)):
        image = imageio.imread(image_path)
        gt = imageio.imread(gt_path)
        gt = (gt > 0)   # binarise the instances

        if task == "boundaries":
            bd = find_boundaries(gt)
            gt = np.stack([gt, bd])

        # HACK: values hard coded for livecell
        prediction = predict_with_halo(
            input_=image, model=model, gpu_ids=[device], block_shape=(512, 512), halo=(64, 64), disable_tqdm=True,
        )

        prediction = prediction.squeeze()
        prediction = (prediction > 0.5)

        visualize = False
        if visualize:
            import napari
            v = napari.Viewer()
            v.add_image(image)
            v.add_labels(gt)
            v.add_labels(prediction)
            napari.run()

        score = dice_score(gt=gt, seg=prediction)
        assert score > 0 and score <= 1
        scores.append(score)

    mean_dice = np.mean(scores)
    print(mean_dice)


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
            name=name, model=model, dataset=dataset, task=task, save_root=save_root, device=device
        )

    elif phase == "predict":
        run_inference(
            name=name, model=model, dataset=dataset, task=task, save_root=save_root, device=device
        )

    else:
        print(f"'{phase}' is not a valid mode. Choose from 'train' / 'predict'.")


if __name__ == "__main__":
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
