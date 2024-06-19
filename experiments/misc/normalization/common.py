import os
from glob import glob

import h5py
import z5py
import numpy as np
import imageio.v3 as imageio

from torch_em.model import UNet2d, UNet3d
from torch_em.data import MinTwoInstanceSampler, datasets

from micro_sam.evaluation.livecell import _get_livecell_paths


# for HLRN
# ROOT = "/scratch/projects/nim00007/sam/data"
# SAVE_DIR = "/scratch/share/cidas/cca/test/verify_normalization"

# for local
ROOT = "/media/anwai/ANWAI/data/"
SAVE_DIR = "/media/anwai/ANWAI/predictions/verify_normalization"


def get_model(dataset, task, norm):
    out_chans = 1 if task == "binary" else 2

    if dataset == "livecell":
        model_choice = "unet2d"
    elif dataset in ["mouse_embryo", "mitoem", "plantseg"]:
        model_choice = "unet3d"
    else:
        raise ValueError

    if model_choice == "unet2d":
        model = UNet2d(
            in_channels=1,
            out_channels=out_chans,
            initial_features=64,
            depth=4,
            final_activation="Sigmoid",
            norm=norm,
        )
    elif model_choice == "unet3d":
        model = UNet3d(
            in_channels=1,
            out_channels=out_chans,
            initial_features=64,
            depth=4,
            final_activation="Sigmoid",
            norm=norm,
        )
    else:
        raise ValueError

    return model


def get_experiment_name(dataset, task, norm, model_choice):
    if model_choice == "unet":
        cfg = "2d" if dataset == "livecell" else "3d"
    else:
        raise ValueError

    name = f"{dataset}_{model_choice}{cfg}_{norm}_{task}"

    return name


def get_dataloaders(dataset, task):
    assert task in ["binary", "boundaries"]

    sampler = MinTwoInstanceSampler()

    if dataset == "livecell":
        train_loader = datasets.get_livecell_loader(
            path=os.path.join(ROOT, "livecell"),
            split="train",
            patch_shape=(512, 512),
            batch_size=2,
            binary=True if task == "binary" else False,
            boundaries=True if task == "boundaries" else False,
            num_workers=16,
            sampler=sampler,
        )
        val_loader = datasets.get_livecell_loader(
            path=os.path.join(ROOT, "livecell"),
            split="val",
            patch_shape=(512, 512),
            batch_size=1,
            binary=True if task == "binary" else False,
            boundaries=True if task == "boundaries" else False,
            num_workers=16,
            sampler=sampler,
        )

    elif dataset == "plantseg":
        train_loader = datasets.get_plantseg_loader(
            path=os.path.join(ROOT, "plantseg"),
            name="root",
            split="train",
            patch_shape=(32, 256, 256),
            batch_size=2,
            binary=True,
            boundaries=True if task == "boundaries" else False,
            num_workers=16,
            sampler=sampler,
        )

        val_loader = datasets.get_plantseg_loader(
            path=os.path.join(ROOT, "plantseg"),
            name="root",
            split="val",
            patch_shape=(32, 256, 256),
            batch_size=1,
            binary=True,
            boundaries=True if task == "boundaries" else False,
            num_workers=16,
            sampler=sampler,
        )

    elif dataset == "mitoem":
        train_loader = datasets.get_mitoem_loader(
            path=os.path.join(ROOT, "mitoem"),
            splits="train",
            patch_shape=(32, 256, 256),
            batch_size=2,
            binary=True if task == "binary" else False,
            boundaries=True if task == "boundaries" else False,
            num_workers=16,
            sampler=sampler,
        )

        val_loader = datasets.get_mitoem_loader(
            path=os.path.join(ROOT, "mitoem"),
            splits="val",
            patch_shape=(32, 256, 256),
            batch_size=1,
            binary=True if task == "binary" else False,
            boundaries=True if task == "boundaries" else False,
            num_workers=16,
            sampler=sampler,
        )

    elif dataset == "mouse_embryo":
        train_loader = datasets.get_mouse_embryo_loader(
            path=os.path.join(ROOT, "mouse-embryo"),
            name="nuclei",
            split="train",
            patch_shape=(32, 256, 256),
            batch_size=2,
            binary=True,
            boundaries=True if task == "boundaries" else False,
            num_workers=16,
            sampler=sampler,
        )

        val_loader = datasets.get_mouse_embryo_loader(
            path=os.path.join(ROOT, "mouse-embryo"),
            name="nuclei",
            split="val",
            patch_shape=(32, 256, 256),
            batch_size=1,
            binary=True,
            boundaries=True if task == "boundaries" else False,
            num_workers=16,
            sampler=sampler,
        )

    else:
        raise ValueError(f"{dataset} is not a valid dataset choice for this experiment.")

    return train_loader, val_loader


def dice_score(gt, seg, eps=1e-7):
    nom = 2 * np.sum(gt * seg)
    denom = np.sum(gt) + np.sum(seg)
    score = float(nom) / float(denom + eps)
    return score


def get_test_images(dataset):
    if dataset == "livecell":
        image_paths, gt_paths = _get_livecell_paths(input_folder=os.path.join(ROOT, "livecell"), split="test")
        return image_paths, gt_paths

    else:
        if dataset == "mouse_embryo":
            volume_paths = sorted(glob(os.path.join(ROOT, "mouse-embryo", "Nuclei", "test", "*.h5")))
        elif dataset == "plantseg":
            volume_paths = sorted(glob(os.path.join(ROOT, "plantseg", "root_test", "*.h5")))
        elif dataset == "mitoem":
            volume_paths = sorted(glob(os.path.join(ROOT, "mitoem", "*_val.n5")))

        return volume_paths, volume_paths


def _load_image(input_path, key=None):
    if key is None:
        image = imageio.imread(input_path)

    else:
        if input_path.endswith(".h5"):
            with h5py.File(input_path, "r") as f:
                image = f[key][:]
        else:
            with z5py.File(input_path, "r") as f:
                image = f[key][:]

    return image
