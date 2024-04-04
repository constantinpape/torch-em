import os
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
import imageio.v3 as imageio

import torch

import torch_em
from torch_em.util import segmentation
from torch_em.model import get_vimunet_model
from torch_em.util.prediction import predict_with_halo
from torch_em.loss import DiceBasedDistanceLoss

from elf.evaluation import mean_segmentation_accuracy

from obtain_lm_datasets import get_lm_loaders


ROOT = "/scratch/projects/nim00007/sam/data"


def run_lm_training(args):
    # the dataloaders for lm datasets
    train_loader, val_loader = get_lm_loaders(ROOT, (512, 512))

    if args.pretrained:
        assert args.model_type == "vim_t"
        checkpoint = "/scratch/usr/nimanwai/models/Vim-tiny/vim_tiny_73p1.pth"
    else:
        checkpoint = None

    # the vision-mamba + decoder (UNet-based) model
    model = get_vimunet_model(
        out_channels=3,
        model_type=args.model_type,
        checkpoint=checkpoint,
        with_cls_token=True
    )

    # saving the model checkpoints
    save_root = os.path.join(
        args.save_root, "pretrained" if args.pretrained else "scratch", "distances", args.model_type
    )

    # loss function
    loss = DiceBasedDistanceLoss(mask_distances_in_bg=True)

    # trainer for the segmentation task
    trainer = torch_em.default_segmentation_trainer(
        name="lm-vimunet",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-5,
        loss=loss,
        metric=loss,
        log_image_interval=50,
        save_root=save_root,
        compile_model=False,
        scheduler_kwargs={"mode": "min", "factor": 0.9, "patience": 10}
    )
    trainer.fit(iterations=args.iterations)


def run_lm_inference(args, device):
    # saving the model checkpoints
    save_root = os.path.join(
        args.save_root, "pretrained" if args.pretrained else "scratch", "distances", args.model_type
    )

    checkpoint = os.path.join(save_root, "checkpoints", "lm-vimunet", "best.pt")

    # the vision-mamba + decoder (UNet-based) model
    model = get_vimunet_model(
        out_channels=3,
        model_type=args.model_type,
        with_cls_token=True,
        checkpoint=checkpoint
    )

    raise NotImplementedError

    res_path = os.path.join(save_root, "results.csv")
    if os.path.exists(res_path) and not args.force:
        print(pd.read_csv(res_path))
        print(f"The result is saved at {res_path}")
        return

    msa_list, sa50_list, sa75_list = [], [], []
    for image_path, label_path in tqdm(zip(all_test_images, all_test_labels), total=len(all_test_images)):
        image = imageio.imread(image_path)
        labels = imageio.imread(label_path)

        predictions = predict_with_halo(
            image, model, [device], block_shape=[512, 512], halo=[128, 128], disable_tqdm=True,
        )

        fg, cdist, bdist = predictions
        instances = segmentation.watershed_from_center_and_boundary_distances(
            cdist, bdist, fg, min_size=50,
            center_distance_threshold=0.5,
            boundary_distance_threshold=0.6,
            distance_smoothing=1.0
        )

        msa, sa_acc = mean_segmentation_accuracy(instances, labels, return_accuracies=True)
        msa_list.append(msa)
        sa50_list.append(sa_acc[0])
        sa75_list.append(sa_acc[5])

    res = {
        "LiveCELL": "Metrics",
        "mSA": np.mean(msa_list),
        "SA50": np.mean(sa50_list),
        "SA75": np.mean(sa75_list)
    }
    df = pd.DataFrame.from_dict([res])
    df.to_csv(res_path)
    print(df)
    print(f"The result is saved at {res_path}")


def main(args):
    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "GPU not available, hence running on CPU")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.train:
        run_lm_training(args)

    if args.predict:
        run_lm_inference(args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=int(1e5))
    parser.add_argument("-s", "--save_root", type=str, default=os.path.join(ROOT, "experiments", "vimunet"))
    parser.add_argument("-m", "--model_type", type=str, default="vim_t")

    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")

    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()
    main(args)
