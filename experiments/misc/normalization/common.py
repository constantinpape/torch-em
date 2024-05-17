import os

from torch_em.model import UNet2d, UNet3d
from torch_em.data import MinTwoInstanceSampler
from torch_em.data.datasets import get_livecell_loader, get_plantseg_loader, get_mitoem_loader

from micro_sam.evaluation.livecell import _get_livecell_paths


ROOT = "/scratch/projects/nim00007/sam/data"


def get_model(dataset, task, norm):
    out_chans = 1 if task == "binary" else 2

    if dataset == "livecell":
        model_choice = "unet2d"
    elif dataset in ["mitoem", "plantseg"]:
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
        train_loader = get_livecell_loader(
            path=os.path.join(ROOT, "livecell"),
            split="train",
            patch_shape=(512, 512),
            batch_size=2,
            binary=True if task == "binary" else False,
            boundaries=True if task == "boundaries" else False,
            num_workers=16,
            sampler=sampler,
        )
        val_loader = get_livecell_loader(
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
        train_loader = get_plantseg_loader(
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

        val_loader = get_plantseg_loader(
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
        train_loader = get_mitoem_loader(
            path=os.path.join(ROOT, "mitoem"),
            splits="train",
            patch_shape=(32, 256, 256),
            batch_size=2,
            binary=True if task == "binary" else False,
            boundaries=True if task == "boundaries" else False,
            num_workers=16,
            sampler=sampler,
        )

        val_loader = get_mitoem_loader(
            path=os.path.join(ROOT, "mitoem"),
            splits="val",
            patch_shape=(32, 256, 256),
            batch_size=1,
            binary=True if task == "binary" else False,
            boundaries=True if task == "boundaries" else False,
            num_workers=16,
            sampler=sampler,
        )

    else:
        raise ValueError(f"{dataset} is not a valid dataset choice for this experiment.")

    return train_loader, val_loader


def get_test_images(dataset):
    if dataset == "livecell":
        image_paths, gt_paths = _get_livecell_paths(input_folder=os.path.join(ROOT, "livecell"), split="test")
        return image_paths, gt_paths

    else:
        raise NotImplementedError
