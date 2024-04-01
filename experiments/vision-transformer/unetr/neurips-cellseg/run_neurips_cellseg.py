import os
import argparse

import torch

import torch_em
from torch_em.model import UNETR, UNet2d
from torch_em.data import MinInstanceSampler
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

MODELS = {
    "vit_t": "/scratch/usr/nimanwai/models/segment-anything/checkpoints/vit_t_mobile_sam.pth",
    "vit_b": "/scratch/usr/nimanwai/models/segment-anything/checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": "/scratch/usr/nimanwai/models/segment-anything/checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": "/scratch/usr/nimanwai/models/segment-anything/checkpoints/sam_vit_h_4b8939.pth"
}


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
        output_channels = len(OFFSETS) + 1

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


def get_model(args, device):
    output_channels = get_output_channels(args)

    if args.model_type == "unet":
        # the UNet model
        model = UNet2d(
            in_channels=3,
            out_channels=output_channels,
            initial_features=64,
            final_activation="Sigmoid",
        )
    else:
        # the UNETR model
        model = UNETR(
            encoder=args.model_type,
            out_channels=output_channels,
            use_sam_stats=args.pretrained,
            encoder_checkpoint=MODELS[args.model_type] if args.pretrained else None,
            final_activation="Sigmoid"
        )
        model.to(device)

    return model


def run_neurips_cellseg_training(args, device):
    # the dataloaders for neurips cellseg dataset
    train_loader, val_loader = get_loaders(args)

    model = get_model(args, device)

    save_root = get_save_root(args)

    # loss function
    loss = get_loss_function(args)

    trainer = torch_em.default_segmentation_trainer(
        name="neurips-cellseg-unet" if args.model_type == "unet" else "neurips-cellseg-unetr",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=1e-5,
        loss=loss,
        metric=loss,
        log_image_interval=50,
        save_root=save_root,
        compile_model=False,
        scheduler_kwargs={"mode": "min", "factor": 0.9, "patience": 10}
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
