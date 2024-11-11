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
from torch_em.loss import DiceLoss, DiceBasedDistanceLoss

from elf.evaluation import mean_segmentation_accuracy


ROOT = "/scratch/usr/nimanwai"


def get_loaders(input, boundaries, distances):
    label_trafo = None
    if distances:
        label_trafo = torch_em.transform.label.PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False, foreground=True, min_size=25,
        )

    kwargs = {
        "path": input,
        "patch_shape": (512, 512),
        "label_dtype": torch.float32,
        "boundaries": boundaries,
        "label_transform": label_trafo,
        "num_workers": 16,
        "download": True,
        "shuffle": True,
    }

    train_loader = get_livecell_loader(split="train", batch_size=2, **kwargs)
    val_loader = get_livecell_loader(split="val", batch_size=1, **kwargs)
    return train_loader, val_loader


def get_output_channels(boundaries):
    if boundaries:
        output_channels = 2
    else:
        output_channels = 3
    return output_channels


def get_loss_function(distances):
    if distances:
        loss = DiceBasedDistanceLoss(mask_distances_in_bg=True)
    else:
        loss = DiceLoss()
    return loss


def get_save_root(boundaries, model_type, save_root):
    # experiment_type
    if boundaries:
        experiment_type = "boundaries"
    else:
        experiment_type = "distances"

    # saving the model checkpoints
    save_root = os.path.join(save_root, "scratch", experiment_type, model_type)
    return save_root


def run_livecell_training(args):
    # the dataloaders for livecell dataset
    train_loader, val_loader = get_loaders(input=args.input, boundaries=args.boundaries, distances=args.distances)
    output_channels = get_output_channels(boundaries=args.boundaries)
    loss = get_loss_function(distances=args.distances)
    save_root = get_save_root(boundaries=args.boundaries, model_type=args.model_type, save_root=args.save_root)

    # the vision-mamba + decoder (UNet-based) model
    model = get_vimunet_model(out_channels=output_channels, model_type=args.model_type, with_cls_token=True)

    # trainer for the segmentation task
    trainer = torch_em.default_segmentation_trainer(
        name="livecell-vimunet",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-4,
        loss=loss,
        metric=loss,
        log_image_interval=50,
        save_root=save_root,
        compile_model=False,
        scheduler_kwargs={"mode": "min", "factor": 0.9, "patience": 10},
        mixed_precision=False,
    )
    trainer.fit(iterations=int(1e5))


def run_livecell_inference(args, device):
    output_channels = get_output_channels(boundaries=args.boundaries)
    save_root = get_save_root(boundaries=args.boundaries, model_type=args.model_type, save_root=args.save_root)
    checkpoint = os.path.join(save_root, "checkpoints", "livecell-vimunet", "best.pt")

    # the vision-mamba + decoder (UNet-based) model
    model = get_vimunet_model(
        out_channels=output_channels, model_type=args.model_type, with_cls_token=True, checkpoint=checkpoint,
    )

    # the splits are provided with the livecell dataset
    # to reproduce the results:
    # run the inference on the entire test datasets as it is.
    test_image_dir = os.path.join(ROOT, "data", "livecell", "images", "livecell_test_images")
    all_test_labels = glob(os.path.join(ROOT, "data", "livecell", "annotations", "livecell_test_images", "*", "*"))

    msa_list, sa50_list, sa75_list = [], [], []
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
        else:
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
        "LIVECell": "Metrics",
        "mSA": np.mean(msa_list),
        "SA50": np.mean(sa50_list),
        "SA75": np.mean(sa75_list)
    }

    res_path = os.path.join(args.result_path, "results.csv")
    df = pd.DataFrame.from_dict([res])
    df.to_csv(res_path)
    print(df)
    print(f"The result is saved at {res_path}")


def main(args):
    assert (args.boundaries + args.distances) == 1, "Choose only one of boundaries / distances to run."

    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "GPU not available, hence running on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        run_livecell_training(args)

    if args.predict:
        run_livecell_inference(args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, default=os.path.join(ROOT, "data", "livecell"), help="Path to LIVECell dataset."
    )
    parser.add_argument(
        "-s", "--save_root", type=str, default="./", help="Path where the model checkpoints will be saved."
    )
    parser.add_argument(
        "-m", "--model_type", type=str, default="vim_t", help="Choice of ViM backbone."
    )
    parser.add_argument(
        "--train", action="store_true", help="Whether to train the model."
    )
    parser.add_argument(
        "--predict", action="store_true", help="Whether to run inference on the trained model."
    )
    parser.add_argument(
        "--result_path", type=str, default="./", help="Path to save quantitative results."
    )
    parser.add_argument(
        "--boundaries", action="store_true", help="Runs the boundary-based methods."
    )
    parser.add_argument(
        "--distances", action="store_true", help="Runs the distance-based methods."
    )
    args = parser.parse_args()
    main(args)
