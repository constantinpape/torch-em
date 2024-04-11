import os
import time
import argparse
from glob import glob

import numpy as np
import pandas as pd
import imageio.v3 as imageio

import torch

import torch_em
from torch_em.util import segmentation
from torch_em.data import ConcatDataset
from torch_em.model import UNETR, UNet2d
from torch_em.transform.raw import standardize
from torch_em.model.unetr import SingleDeconv2DBlock
from torch_em.util.prediction import predict_with_padding
from torch_em.data.datasets import get_livecell_loader, get_livecell_dataset
from torch_em.loss import DiceLoss, LossWrapper, ApplyAndRemoveMask, DiceBasedDistanceLoss

from elf.evaluation import mean_segmentation_accuracy


ROOT = "/scratch/usr/nimanwai"

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


def get_loaders(args, patch_shape=(512, 512), n_samples=25):
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

    # let's get 25 images per cell-type for training
    all_tr_ds = []
    cell_types = ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3"]
    for ctype in cell_types:
        all_tr_ds.append(
            get_livecell_dataset(
                path=args.input,
                split="train",
                patch_shape=patch_shape,
                label_dtype=torch.float32,
                boundaries=args.boundaries,
                label_transform=label_trafo,
                offsets=OFFSETS if args.affinities else None,
                n_samples=n_samples,
                cell_types=[ctype]
            )
        )

    train_dataset = ConcatDataset(*all_tr_ds)
    train_loader = torch_em.get_data_loader(train_dataset, shuffle=True, batch_size=2, num_workers=16)

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

    model_name = args.model_type

    # saving the model checkpoints
    save_root = os.path.join(
        args.save_root,
        f"{args.lr}",
        "pretrained" if args.pretrained else "scratch",
        experiment_type,
        model_name
    )
    return save_root


def get_model(args, device):
    output_channels = get_output_channels(args)

    if args.model_type == "unet":
        # the UNet model
        model = UNet2d(
            in_channels=1,
            out_channels=output_channels,
            initial_features=64,
            final_activation="Sigmoid",
            sampler_impl=SingleDeconv2DBlock
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


def run_livecell_unetr_training(args, device):
    model = get_model(args, device)

    save_root = get_save_root(args)

    # loss function
    loss = get_loss_function(args)

    # the dataloaders for livecell dataset
    train_loader, val_loader = get_loaders(args)

    trainer = torch_em.default_segmentation_trainer(
        name="livecell-per25-unet" if args.model_type == "unet" else "livecell-per25-unetr",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        loss=loss,
        metric=loss,
        log_image_interval=50,
        save_root=save_root,
        compile_model=False,
        scheduler_kwargs={"mode": "min", "factor": 0.9, "patience": 50}
    )

    trainer.fit(args.iterations)


def run_livecell_unetr_inference(args, device):
    save_root = get_save_root(args)

    checkpoint = os.path.join(
        save_root,
        "checkpoints",
        "livecell-per25-unet" if args.model_type == "unet" else "livecell-per25-unetr",
        "best.pt"
    )

    model = get_model(args, device)

    assert os.path.exists(checkpoint), checkpoint
    model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu'))["model_state"])
    model.to(device)
    model.eval()

    test_image_dir = os.path.join(ROOT, "data", "livecell", "images", "livecell_test_images")
    all_test_labels = glob(os.path.join(ROOT, "data", "livecell", "annotations", "livecell_test_images", "*", "*"))

    res_path = os.path.join(save_root, "results.csv")
    if os.path.exists(res_path) and not args.force:
        print(pd.read_csv(res_path))
        print(f"The result is saved at {res_path}")
        return

    msa_list, sa50_list, sa75_list = [], [], []
    all_times = []
    for i, label_path in enumerate(all_test_labels):
        labels = imageio.imread(label_path)
        image_id = os.path.split(label_path)[-1]

        image = imageio.imread(os.path.join(test_image_dir, image_id))
        image = standardize(image)

        if i == 0:  # dry run to load all modules to avoid timing errors for the first iteration
            predictions = predict_with_padding(model, image, min_divisible=(16, 16), device=device)

        start = time.time()
        predictions = predict_with_padding(model, image, min_divisible=(16, 16), device=device)
        end = time.time()
        diff = end - start
        all_times.append(diff)

        predictions = predictions.squeeze()

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

    # let' track the timings and report them
    print("Mean:", np.mean(all_times))
    print("Standard deviation:", np.std(all_times))


def main(args):
    assert (args.boundaries + args.affinities + args.distances) == 1

    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "GPU not available, hence running on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        run_livecell_unetr_training(args, device)

    if args.predict:
        run_livecell_unetr_inference(args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=os.path.join(ROOT, "data", "livecell"))
    parser.add_argument("--iterations", type=int, default=int(1e5))
    parser.add_argument("-s", "--save_root", type=str, default=os.path.join(ROOT, "experiments", "vimunet-limited"))
    parser.add_argument("-m", "--model_type", type=str, required=True)
    parser.add_argument("--lr", type=float, default=1e-5)

    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--force", action="store_true")

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")

    parser.add_argument("--boundaries", action="store_true")
    parser.add_argument("--affinities", action="store_true")
    parser.add_argument("--distances", action="store_true")

    args = parser.parse_args()
    main(args)
