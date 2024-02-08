import os
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import imageio.v3 as imageio

import torch

import torch_em
from torch_em.util import segmentation
from torch_em.model import get_vimunet_model
from torch_em.transform.raw import standardize
from torch_em.data.datasets import get_livecell_loader
from torch_em.loss import DiceLoss, LossWrapper, ApplyAndRemoveMask, DiceBasedDistanceLoss

from elf.evaluation import mean_segmentation_accuracy


ROOT = "/scratch/usr/nimanwai"

OFFSETS = [
    [-1, 0], [0, -1],
    [-3, 0], [0, -3],
    [-9, 0], [0, -9],
    [-27, 0], [0, -27]
]


def get_loaders(args, patch_shape=(520, 704)):
    if args.distances:
        label_trafo = torch_em.transform.label.PerObjectDistanceTransform(
            distances=True,
            boundary_distances=True,
            directed_distances=False,
            foreground=True,
            min_size=25
        )
    else:
        label_trafo = None

    train_loader = get_livecell_loader(
        path=args.input,
        split="train",
        patch_shape=patch_shape,
        batch_size=2,
        label_dtype=torch.float32,
        boundaries=args.boundaries,
        label_transform=label_trafo,
        offsets=OFFSETS if args.affinities else None,
        num_workers=16
    )

    val_loader = get_livecell_loader(
        path=args.input,
        split="val",
        patch_shape=patch_shape,
        batch_size=1,
        label_dtype=torch.float32,
        boundaries=args.boundaries,
        label_transform=label_trafo,
        offsets=OFFSETS if args.affinities else None,
        num_workers=16
    )

    return train_loader, val_loader


def get_output_channels(args):
    if args.boundaries:
        output_channels = 2
    elif args.distances:
        output_channels = 3
    elif args.affinities:
        output_channels = (len(OFFSETS) + 1)

    return output_channels


def get_loss_function(args):
    if args.affinities:
        loss = LossWrapper(
            loss=DiceLoss(),
            transform=ApplyAndRemoveMask(masking_method="multiply")
        )
    elif args.distances:
        loss = DiceBasedDistanceLoss(mask_distances_in_bg=True)

    else:
        loss = DiceLoss()

    return loss


def get_save_root(args):
    # experiment_type
    if args.boundaries:
        experiment_type = "boundaries"
    elif args.affinities:
        experiment_type = "affinities"
    elif args.distances:
        experiment_type = "distances"
    else:
        raise ValueError

    # saving the model checkpoints
    save_root = os.path.join(
        args.save_root,
        "pretrained" if args.pretrained else "scratch",
        experiment_type,
        args.model_type
    )

    return save_root


def run_livecell_training(args):
    # the dataloaders for livecell dataset
    train_loader, val_loader = get_loaders(args)

    if args.pretrained:
        checkpoint = "/scratch/usr/nimanwai/models/Vim-tiny/vim_tiny_73p1.pth"
    else:
        checkpoint = None

    output_channels = get_output_channels(args)

    # the vision-mamba + decoder (UNet-based) model
    model = get_vimunet_model(
        out_channels=output_channels,
        model_type=args.model_type,
        checkpoint=checkpoint,
        with_cls_token=True,
        use_conv_transpose=args.use_conv_transpose
    )

    print(model.decoder.samplers)

    save_root = get_save_root(args)

    # loss function
    loss = get_loss_function(args)

    name = "livecell-vimunet"
    if args.use_conv_transpose:
        name += "-conv-transpose"
    else:
        name += "-bilinear"

    # trainer for the segmentation task
    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-4,
        loss=loss,
        metric=loss,
        log_image_interval=50,
        save_root=save_root,
        compile_model=False
    )
    trainer.fit(iterations=int(args.iterations))


def run_livecell_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_channels = get_output_channels(args)

    save_root = get_save_root(args)
    checkpoint = os.path.join(save_root, "checkpoints", "livecell-vimunet", "best.pt")

    # the vision-mamba + decoder (UNet-based) model
    model = get_vimunet_model(
        out_channels=output_channels,
        model_type=args.model_type,
        with_cls_token=True,
        checkpoint=checkpoint,
        use_conv_transpose=args.use_conv_transpose
    )

    test_image_dir = os.path.join(ROOT, "data", "livecell", "images", "livecell_test_images")
    all_test_labels = glob(os.path.join(ROOT, "data", "livecell", "annotations", "livecell_test_images", "*", "*"))

    res_path = os.path.join(save_root, "results.csv")
    if os.path.exists(res_path):
        print(pd.read_csv(res_path))
        return

    msa_list, sa50_list = [], []

    for label_path in tqdm(all_test_labels):
        labels = imageio.imread(label_path)
        image_id = os.path.split(label_path)[-1]

        image = imageio.imread(os.path.join(test_image_dir, image_id))
        image = standardize(image)

        tensor_image = torch.from_numpy(image)[None, None].to(device)

        predictions = model(tensor_image)
        predictions = predictions.squeeze().detach().cpu().numpy()

        if args.boundaries:
            fg, bd = predictions
            instances = segmentation.watershed_from_components(bd, fg)

        elif args.affinities:
            fg, affs = predictions[0], predictions[1:]
            instances = segmentation.mutex_watershed_segmentation(fg, affs, offsets=OFFSETS)

        elif args.distances:
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

    res = {
        "LiveCELL": "Metrics",
        "mSA": np.mean(msa_list),
        "SA50": np.mean(sa50_list)
    }
    df = pd.DataFrame.from_dict([res])
    df.to_csv(res_path)
    print(df)
    print(f"The result is saved at {res_path}")


def main(args):
    assert (args.boundaries + args.affinities + args.distances) == 1

    if args.train:
        run_livecell_training(args)

    if args.predict:
        run_livecell_inference(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=os.path.join(ROOT, "data", "livecell"))
    parser.add_argument("--iterations", type=int, default=1e5)
    parser.add_argument("-s", "--save_root", type=str, default=os.path.join(ROOT, "experiments", "vision-mamba"))
    parser.add_argument("-m", "--model_type", type=str, default="vim_t")

    parser.add_argument("--use_conv_transpose", action="store_true")
    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")

    parser.add_argument("--boundaries", action="store_true")
    parser.add_argument("--affinities", action="store_true")
    parser.add_argument("--distances", action="store_true")

    args = parser.parse_args()
    main(args)
