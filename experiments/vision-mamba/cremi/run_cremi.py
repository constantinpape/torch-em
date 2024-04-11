import os
import argparse
import numpy as np
import pandas as pd
from glob import glob

import imageio.v3 as imageio

import torch

import torch_em
from torch_em.util import segmentation
from torch_em.data import MinInstanceSampler
from torch_em.model import get_vimunet_model
from torch_em.data.datasets import get_cremi_loader
from torch_em.util.prediction import predict_with_halo
from torch_em.loss import DiceLoss, DiceBasedDistanceLoss

from elf.evaluation import mean_segmentation_accuracy


ROOT = "/scratch/usr/nimanwai"
CREMI_TEST_ROOT = "/scratch/projects/nim00007/sam/data/cremi/slices_original"


def get_loaders(args, patch_shape=(1, 512, 512)):
    if args.distances:
        label_trafo = torch_em.transform.label.PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False, foreground=True
        )
    else:
        label_trafo = None

    train_rois = {"A": np.s_[0:75, :, :], "B": np.s_[0:75, :, :], "C": np.s_[0:75, :, :]}
    val_rois = {"A": np.s_[75:100, :, :], "B": np.s_[75:100, :, :], "C": np.s_[75:100, :, :]}

    sampler = MinInstanceSampler()

    train_loader = get_cremi_loader(
        path=args.input,
        patch_shape=patch_shape,
        batch_size=2,
        label_transform=label_trafo,
        rois=train_rois,
        sampler=sampler,
        ndim=2,
        label_dtype=torch.float32,
        defect_augmentation_kwargs=None,
        boundaries=args.boundaries,
        num_workers=16,
    )
    val_loader = get_cremi_loader(
        path=args.input,
        patch_shape=patch_shape,
        batch_size=1,
        label_transform=label_trafo,
        rois=val_rois,
        sampler=sampler,
        ndim=2,
        label_dtype=torch.float32,
        defect_augmentation_kwargs=None,
        boundaries=args.boundaries,
        num_workers=16,
    )

    return train_loader, val_loader


def get_output_channels(args):
    if args.distances:
        output_channels = 3
    else:
        output_channels = 1

    return output_channels


def get_loss_function(args):
    if args.distances:
        loss = DiceBasedDistanceLoss(mask_distances_in_bg=True)
    else:
        loss = DiceLoss()

    return loss


def get_save_root(args):
    if args.boundaries:
        experiment_type = "boundaries"
    else:
        experiment_type = "distances"

    model_name = args.model_type

    # saving the model checkpoints
    save_root = os.path.join(args.save_root, "scratch", experiment_type, model_name)
    return save_root


def run_cremi_training(args):
    # the dataloaders for cremi dataset
    train_loader, val_loader = get_loaders(args)

    output_channels = get_output_channels(args)

    # the vision-mamba + decoder (UNet-based) model
    model = get_vimunet_model(
        out_channels=output_channels,
        model_type=args.model_type,
        with_cls_token=True
    )

    save_root = get_save_root(args)

    # loss function
    loss = get_loss_function(args)

    # trainer for the segmentation task
    trainer = torch_em.default_segmentation_trainer(
        name="cremi-vimunet",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-4,
        loss=loss,
        metric=loss,
        log_image_interval=50,
        save_root=save_root,
        compile_model=False,
        scheduler_kwargs={"mode": "min", "factor": 0.9, "patience": 10}
    )
    trainer.fit(iterations=1e5)


def run_cremi_inference(args, device):
    output_channels = get_output_channels(args)

    save_root = get_save_root(args)

    checkpoint = os.path.join(save_root, "checkpoints", "cremi-vimunet", "best.pt")

    # the vision-mamba + decoder (UNet-based) model
    model = get_vimunet_model(
        out_channels=output_channels,
        model_type=args.model_type,
        with_cls_token=True,
        checkpoint=checkpoint
    )

    all_test_images = glob(os.path.join(CREMI_TEST_ROOT, "raw", "cremi_test_sampleC*.tif"))
    all_test_labels = glob(os.path.join(CREMI_TEST_ROOT, "labels", "cremi_test_sampleC*.tif"))

    res_path = os.path.join(save_root, "results.csv")
    if os.path.exists(res_path) and not args.force:
        print(pd.read_csv(res_path))
        print(f"The result is saved at {res_path}")
        return

    msa_list, sa50_list, sa75_list = [], [], []
    for image_path, label_path in zip(all_test_images, all_test_labels):
        image = imageio.imread(image_path)
        labels = imageio.imread(label_path)

        predictions = predict_with_halo(
            image, model, [device], block_shape=[512, 512], halo=[128, 128], disable_tqdm=True,
        )

        if args.boundaries:
            bd = predictions.squeeze()
            instances = segmentation.watershed_from_components(bd, np.ones_like(bd))

        else:
            fg, cdist, bdist = predictions
            instances = segmentation.watershed_from_center_and_boundary_distances(
                cdist, bdist, np.ones_like(fg), min_size=50,
                center_distance_threshold=0.5,
                boundary_distance_threshold=0.6,
                distance_smoothing=1.0
            )

        msa, sa_acc = mean_segmentation_accuracy(instances, labels, return_accuracies=True)
        msa_list.append(msa)
        sa50_list.append(sa_acc[0])
        sa75_list.append(sa_acc[5])

    res = {
        "LIVECell": "Metrics",
        "mSA": np.mean(msa_list),
        "SA50": np.mean(sa50_list),
        "SA75": np.mean(sa75_list)
    }
    df = pd.DataFrame.from_dict([res])
    df.to_csv(res_path)
    print(df)
    print(f"The result is saved at {res_path}")


def main(args):
    assert (args.boundaries + args.distances) == 1

    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "GPU not available, hence running on CPU")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.train:
        run_cremi_training(args)

    if args.predict:
        run_cremi_inference(args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=os.path.join(ROOT, "data", "cremi"))
    parser.add_argument("-s", "--save_root", type=str, default=None)
    parser.add_argument("-m", "--model_type", type=str, default="vim_t")

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")

    parser.add_argument("--force", action="store_true")

    parser.add_argument("--boundaries", action="store_true")
    parser.add_argument("--distances", action="store_true")

    args = parser.parse_args()
    main(args)
