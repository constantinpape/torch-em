import os
from glob import glob

import h5py
import z5py
import numpy as np
import imageio.v3 as imageio

from torch_em.model import UNet2d, UNet3d
from torch_em.data import MinTwoInstanceSampler, datasets
from torch_em.transform.raw import normalize_percentile, get_default_mean_teacher_augmentations

from micro_sam.evaluation.livecell import _get_livecell_paths


if os.path.exists("/scratch/share/cidas"):
    ROOT = "/scratch/share/cidas/cca/data"
    SAVE_DIR = "/scratch/share/cidas/cca/test/verify_normalization"
else:
    ROOT = "/media/anwai/ANWAI/data/"
    SAVE_DIR = "/media/anwai/ANWAI/test/verify_normalization"


def get_model(dataset, task, norm):
    out_chans = 1 if task == "binary" or dataset == "plantseg" else 2
    _model_class = UNet2d if dataset == "livecell" else UNet3d

    model = _model_class(
        in_channels=1,
        out_channels=out_chans,
        initial_features=64,
        depth=4,
        final_activation="Sigmoid",
        norm=norm,
    )
    return model


def get_experiment_name(dataset, task, norm, model_choice):
    cfg = "2d" if dataset == "livecell" else "3d"
    name = f"{dataset}_{model_choice}{cfg}_{norm}_{task}"
    return name


def get_dataloaders(dataset, task):
    assert task in ["binary", "boundaries"]
    sampler = MinTwoInstanceSampler()

    loader_kwargs = {
        "num_workers": 16, "download": True, "sampler": sampler,
        "raw_transform": get_default_mean_teacher_augmentations(p=0.3, norm=normalize_percentile),
    }

    if dataset == "livecell":
        train_loader = datasets.get_livecell_loader(
            path=os.path.join(ROOT, "livecell"),
            split="train",
            patch_shape=(512, 512),
            batch_size=2,
            binary=True if task == "binary" else False,
            boundaries=True if task == "boundaries" else False,
            **loader_kwargs
        )
        val_loader = datasets.get_livecell_loader(
            path=os.path.join(ROOT, "livecell"),
            split="val",
            patch_shape=(512, 512),
            batch_size=1,
            binary=True if task == "binary" else False,
            boundaries=True if task == "boundaries" else False,
            **loader_kwargs
        )
    elif dataset == "plantseg":
        train_loader = datasets.get_plantseg_loader(
            path=os.path.join(ROOT, "plantseg"),
            name="root",
            split="train",
            patch_shape=(32, 512, 512),
            batch_size=2,
            boundaries=True if task == "boundaries" else False,
            **loader_kwargs
        )

        val_loader = datasets.get_plantseg_loader(
            path=os.path.join(ROOT, "plantseg"),
            name="root",
            split="test",
            patch_shape=(32, 512, 512),
            batch_size=1,
            boundaries=True if task == "boundaries" else False,
            **loader_kwargs
        )
    elif dataset == "mitoem":
        train_loader = datasets.get_mitoem_loader(
            path=os.path.join(ROOT, "mitoem"),
            splits="train",
            patch_shape=(32, 512, 512),
            batch_size=2,
            binary=True if task == "binary" else False,
            boundaries=True if task == "boundaries" else False,
            **loader_kwargs
        )

        val_loader = datasets.get_mitoem_loader(
            path=os.path.join(ROOT, "mitoem"),
            splits="val",
            patch_shape=(32, 512, 512),
            batch_size=1,
            binary=True if task == "binary" else False,
            boundaries=True if task == "boundaries" else False,
            **loader_kwargs
        )
    elif dataset == "gonuclear":
        train_loader = datasets.get_gonuclear_loader(
            path=os.path.join(ROOT, "gonuclear"),
            patch_shape=(32, 512, 512),
            batch_size=2,
            segmentation_task="nuclei",
            binary=True if task == "binary" else False,
            boundaries=True if task == "boundaries" else False,
            sample_ids=[1135, 1136, 1137],
            **loader_kwargs
        )
        val_loader = datasets.get_gonuclear_loader(
            path=os.path.join(ROOT, "gonuclear"),
            patch_shape=(32, 512, 512),
            batch_size=1,
            segmentation_task="nuclei",
            binary=True if task == "binary" else False,
            boundaries=True if task == "boundaries" else False,
            sample_ids=[1139],
            **loader_kwargs
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
        if dataset == "gonuclear":
            volume_paths = [os.path.join(ROOT, "gonuclear", "gonuclear_datasets", "1170.h5")]
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
