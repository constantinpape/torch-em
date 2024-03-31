import os
import argparse

import torch

import torch_em
from torch_em.data import MinInstanceSampler
from torch_em.model import get_vimunet_model
from torch_em.data.datasets import get_neurips_cellseg_supervised_loader
from torch_em.transform.label import AffinityTransform, BoundaryTransform
from torch_em.loss import DiceLoss, LossWrapper, ApplyAndRemoveMask, DiceBasedDistanceLoss


ROOT = "/scratch/usr/nimanwai/"

OFFSETS = [
    [-1, 0], [0, -1],
    [-3, 0], [0, -3],
    [-9, 0], [0, -9],
    [-27, 0], [0, -27]
]


def get_loaders(args, patch_shape=(512, 512)):
    if args.boundaries:
        label_trafo = BoundaryTransform(add_binary_target=True)
    elif args.affinities:
        label_trafo = AffinityTransform(offsets=OFFSETS, add_binary_target=True, add_mask=True)
    elif args.distances:
        label_trafo = torch_em.transform.label.PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False, foreground=True
        )

    sampler = MinInstanceSampler(min_num_instances=3)

    train_loader = get_neurips_cellseg_supervised_loader(
        root=ROOT,
        split="train",
        patch_shape=patch_shape,
        batch_size=2,
        make_rgb=True,
        download=True,
        label_transform=label_trafo,
        num_workers=16,
        sampler=sampler,
    )
    val_loader = get_neurips_cellseg_supervised_loader(
        root=ROOT,
        split="val",
        patch_shape=patch_shape,
        batch_size=1,
        make_rgb=True,
        download=True,
        label_transform=label_trafo,
        num_workers=16,
        sampler=sampler,
    )

    # increasing the sampling attempts for the neurips cellseg dataset
    train_loader.dataset.max_sampling_attempts = 5000
    val_loader.dataset.max_sampling_attempts = 5000

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

    model_name = args.model_type

    # saving the model checkpoints
    save_root = os.path.join(
        args.save_root, "pretrained" if args.pretrained else "scratch", experiment_type, model_name
    )
    return save_root


def run_neurips_cellseg_training(args, device):
    # the dataloaders for neurips cellseg dataset
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
        with_cls_token=True
    )

    save_root = get_save_root(args)

    # loss function
    loss = get_loss_function(args)

    trainer = torch_em.default_segmentation_trainer(
        name="neurips-cellseg-vimunet",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=1e-4,
        loss=loss,
        metric=loss,
        log_image_interval=50,
        save_root=save_root,
        compile_model=False
    )

    trainer.fit(args.iterations)


def run_neurips_cellseg_inference(args, device):
    raise NotImplementedError


def main(args):
    assert (args.boundaries + args.affinities + args.distances) == 1

    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "GPU not available, hence running on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        run_neurips_cellseg_training(args, device)

    if args.predict:
        run_neurips_cellseg_inference(args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, default=os.path.join(ROOT, "data", "neurips-cell-seg", "zenodo")
    )
    parser.add_argument("--iterations", type=int, default=int(1e5))
    parser.add_argument("-s", "--save_root", type=str, default=os.path.join(ROOT, "experiments", "vimunet"))
    parser.add_argument("-m", "--model_type", type=str, required=True)

    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--force", action="store_true")

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")

    parser.add_argument("--boundaries", action="store_true")
    parser.add_argument("--affinities", action="store_true")
    parser.add_argument("--distances", action="store_true")

    args = parser.parse_args()
    main(args)
