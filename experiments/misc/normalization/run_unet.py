import os
from tqdm import tqdm

import numpy as np

import torch

import torch_em
from torch_em.loss import DiceLoss
from torch_em.util.prediction import predict_with_halo
from torch_em.util.segmentation import watershed_from_components

from elf.evaluation import mean_segmentation_accuracy

from common import get_dataloaders, get_model, get_experiment_name, get_test_images, dice_score, _load_image


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

    dsc_list, msa_list, sa50_list = [], [], []
    for image_path, gt_path in tqdm(zip(image_paths, gt_paths), desc="Predicting", total=len(image_paths)):
        if dataset == "livecell":
            image = _load_image(image_path)
            gt = _load_image(gt_path)
        elif dataset in ["mouse_embryo", "plantseg"]:
            image = _load_image(image_path, "raw")
            gt = _load_image(image_path, "label")

        if dataset == "livecell":
            tile, halo = (512, 512), (64, 64)
        else:
            tile, halo = (64, 256, 256), (16, 64, 64)

        prediction = predict_with_halo(
            input_=image, model=model, gpu_ids=[device], block_shape=tile, halo=halo, disable_tqdm=True,
        )

        prediction = prediction.squeeze()

        if task == "boundaries":
            fg, bd = prediction
            instances = watershed_from_components(boundaries=bd, foreground=fg)

            msa, sa = mean_segmentation_accuracy(segmentation=instances, groundtruth=gt, return_accuracies=True)
            msa_list.append(msa)
            sa50_list.append(sa[0])

        else:
            gt = (gt > 0)   # binarise the instances
            prediction = (prediction > 0.5)  # threshold the predictions

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
