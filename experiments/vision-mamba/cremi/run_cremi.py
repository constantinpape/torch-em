import os
import argparse
import numpy as np
import pandas as pd
from glob import glob

import imageio.v3 as imageio

import torch

import torch_em
from torch_em.loss import DiceLoss
from torch_em.util import segmentation
from torch_em.data import MinInstanceSampler
from torch_em.model import get_vimunet_model
from torch_em.data.datasets import get_cremi_loader
from torch_em.util.prediction import predict_with_halo

from elf.evaluation import mean_segmentation_accuracy


ROOT = "/scratch/usr/nimanwai"

# the splits have been customed made
# to reproduce the results:
# create slices ranging from "100 to 125" for all three volumes
CREMI_TEST_ROOT = "/scratch/projects/nim00007/sam/data/cremi/slices_original"


def get_loaders(args, patch_shape=(1, 512, 512)):
    train_rois = {"A": np.s_[0:75, :, :], "B": np.s_[0:75, :, :], "C": np.s_[0:75, :, :]}
    val_rois = {"A": np.s_[75:100, :, :], "B": np.s_[75:100, :, :], "C": np.s_[75:100, :, :]}

    sampler = MinInstanceSampler()

    train_loader = get_cremi_loader(
        path=args.input,
        patch_shape=patch_shape,
        batch_size=2,
        rois=train_rois,
        sampler=sampler,
        ndim=2,
        label_dtype=torch.float32,
        defect_augmentation_kwargs=None,
        boundaries=True,
        num_workers=16,
    )
    val_loader = get_cremi_loader(
        path=args.input,
        patch_shape=patch_shape,
        batch_size=1,
        rois=val_rois,
        sampler=sampler,
        ndim=2,
        label_dtype=torch.float32,
        defect_augmentation_kwargs=None,
        boundaries=True,
        num_workers=16,
    )
    return train_loader, val_loader


def run_cremi_training(args):
    # the dataloaders for cremi dataset
    train_loader, val_loader = get_loaders(args)

    # the vision-mamba + decoder (UNet-based) model
    model = get_vimunet_model(
        out_channels=1,
        model_type=args.model_type,
        with_cls_token=True
    )

    save_root = os.path.join(args.save_root, "scratch", "boundaries", args.model_type)

    # loss function
    loss = DiceLoss()

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
    save_root = os.path.join(args.save_root, "scratch", "boundaries", args.model_type)
    checkpoint = os.path.join(save_root, "checkpoints", "cremi-vimunet", "best.pt")

    # the vision-mamba + decoder (UNet-based) model
    model = get_vimunet_model(
        out_channels=1,
        model_type=args.model_type,
        with_cls_token=True,
        checkpoint=checkpoint
    )

    all_test_images = glob(os.path.join(CREMI_TEST_ROOT, "raw", "cremi_test_*.tif"))
    all_test_labels = glob(os.path.join(CREMI_TEST_ROOT, "labels", "cremi_test_*.tif"))

    msa_list, sa50_list, sa75_list = [], [], []
    for image_path, label_path in zip(all_test_images, all_test_labels):
        image = imageio.imread(image_path)
        labels = imageio.imread(label_path)

        predictions = predict_with_halo(
            image, model, [device], block_shape=[512, 512], halo=[128, 128], disable_tqdm=True,
        )

        bd = predictions.squeeze()
        instances = segmentation.watershed_from_components(bd, np.ones_like(bd))

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
    res_path = os.path.join("./results.csv")
    df = pd.DataFrame.from_dict([res])
    df.to_csv(res_path)
    print(df)
    print(f"The result is saved at {res_path}")


def main(args):
    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "GPU not available, hence running on CPU")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.train:
        run_cremi_training(args)

    if args.predict:
        run_cremi_inference(args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, default=os.path.join(ROOT, "data", "cremi"), help="Path to CREMI dataset."
    )
    parser.add_argument(
        "-s", "--save_root", type=str, default=None, help="Path where the model checkpoints will be saved."
    )
    parser.add_argument("-m", "--model_type", type=str, default="vim_t", path="Choice of ViM backbone")

    parser.add_argument(
        "--train", action="store_true", help="Whether to train the model."
    )
    parser.add_argument(
        "--predict", action="store_true", help="Whether to run inference on the trained model."
    )
    parser.add_argument(
        "--boundaries", action="store_true", help="Runs the boundary-based methods"
    )
    args = parser.parse_args()
    main(args)
