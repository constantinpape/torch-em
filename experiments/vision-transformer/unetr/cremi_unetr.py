import os
import argparse
import numpy as np

import torch
import torch_em
from torch_em.model.unetr import build_unetr_with_sam_initialization
from torch_em.data.datasets import get_cremi_loader


def do_unetr_training(data_path: str, save_root: str, iterations: int, device, patch_shape=(1, 512, 512)):
    os.makedirs(data_path, exist_ok=True)

    cremi_train_rois = {"A": np.s_[0:75, :, :], "B": np.s_[0:75, :, :], "C": np.s_[0:75, :, :]}
    cremi_val_rois = {"A": np.s_[75:100, :, :], "B": np.s_[75:100, :, :], "C": np.s_[75:100, :, :]}

    train_loader = get_cremi_loader(
        path=data_path,
        patch_shape=patch_shape, download=True,
        rois=cremi_train_rois,
        ndim=2,
        defect_augmentation_kwargs=None,
        boundaries=True,
        batch_size=2
    )

    val_loader = get_cremi_loader(
        path=data_path,
        patch_shape=patch_shape, download=True,
        rois=cremi_val_rois,
        ndim=2,
        defect_augmentation_kwargs=None,
        boundaries=True,
        batch_size=1
    )

    model = build_unetr_with_sam_initialization(
        checkpoint_path="/scratch/usr/nimanwai/models/segment-anything/checkpoints/sam_vit_b_01ec64.pth"
    )
    model.to(device)

    trainer = torch_em.default_segmentation_trainer(
        name="unetr-cremi",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=1e-5,
        log_image_interval=10,
        save_root=save_root,
        compile_model=False
    )

    trainer.fit(iterations)


def main(args):
    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "GPU not available, hence running on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        print("Training a 2D UNETR on Cremi dataset")
        do_unetr_training(
            data_path=args.inputs,
            save_root=args.save_root,
            iterations=args.iterations,
            device=device
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true', help="Enables UNETR training on Cremi dataset")
    parser.add_argument("-i", "--inputs", type=str, default="./cremi/",
                        help="Path where the dataset already exists/will be downloaded by the dataloader")
    parser.add_argument("-s", "--save_root", type=str, default=None,
                        help="Path where checkpoints and logs will be saved")
    parser.add_argument("--iterations", type=int, default=100000, help="No. of iterations to run the training for")
    args = parser.parse_args()
    main(args)
