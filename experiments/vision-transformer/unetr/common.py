import os
import h5py
import argparse
import numpy as np
from typing import Tuple

import imageio.v3 as imageio

from torch_em.util import segmentation
from torch_em.transform.raw import standardize
from torch_em.data.datasets import get_livecell_loader
from torch_em.loss import DiceLoss, LossWrapper, ApplyAndRemoveMask
from torch_em.util.prediction import predict_with_halo, predict_with_padding

import common


OFFSETS = [
    [-1, 0], [0, -1],
    [-3, 0], [0, -3],
    [-9, 0], [0, -9],
    [-27, 0], [0, -27]
]


CELL_TYPES = ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3"]


#
# LIVECELL DATALOADERS
#


def _get_output_channels(with_affinities):
    if with_affinities:
        n_out = len(OFFSETS) + 1
    else:
        n_out = 2
    return n_out


def get_my_livecell_loaders(
        input_path: str,
        patch_shape: Tuple[int, int],
        cell_types: str,
        with_affinities: bool = False
):
    """Returns the LIVECell training and validation dataloaders
    """
    if with_affinities:
        # this returns dataloaders with affinity channels and foreground-background channels
        train_loader = get_livecell_loader(
            path=input_path, split="train", patch_shape=patch_shape, batch_size=2,
            cell_types=[cell_types], download=True, offsets=OFFSETS, num_workers=16
        )
        val_loader = get_livecell_loader(
            path=input_path, split="val", patch_shape=patch_shape, batch_size=1,
            cell_types=[cell_types], download=True, offsets=OFFSETS, num_workers=16
        )

    else:
        # this returns dataloaders with foreground and boundary channels
        train_loader = get_livecell_loader(
            path=input_path, split="train", patch_shape=patch_shape, batch_size=2,
            cell_types=[cell_types], download=True, boundaries=True, num_workers=16
        )
        val_loader = get_livecell_loader(
            path=input_path, split="val", patch_shape=patch_shape, batch_size=1,
            cell_types=[cell_types], download=True, boundaries=True, num_workers=16
        )

    return train_loader, val_loader


#
# UNETR MODEL(S) FROM MONAI AND torch_em
#


def get_unetr_model(
        model_name: str,
        source_choice: str,
        patch_shape: Tuple[int, int],
        sam_initialization: bool,
        output_channels: int
):
    """Returns the expected UNETR model
    """
    if source_choice == "torch-em":
        # this returns the unetr model whihc uses the vision transformer from segment anything
        from torch_em import model as torch_em_models
        model = torch_em_models.UNETR(
            encoder=model_name, out_channels=output_channels,
            encoder_checkpoint_path="/scratch/usr/nimanwai/models/segment-anything/checkpoints/sam_vit_b_01ec64.pth" if sam_initialization else None
        )

    elif source_choice == "monai":
        # this returns the unetr model from monai
        from monai.networks import nets as monai_models
        model = monai_models.unetr.UNETR(
            in_channels=1,
            out_channels=output_channels,
            img_size=patch_shape,
            spatial_dims=2
        )
        model.out_channels = 2  # type: ignore

    else:
        raise ValueError(f"The available UNETR models are either from \"torch-em\" or \"monai\", choose from them instead of - {source_choice}")

    return model


#
# LIVECELL UNETR INFERENCE - foreground boundary / foreground affinities
#

def predict_for_unetr(img_path, model, root_save_dir, ctype, device, with_affinities):
    input_ = imageio.imread(img_path)
    input_ = standardize(input_)

    if with_affinities:  # inference using affinities
        outputs = predict_with_padding(model, input_, device=device, min_divisible=(16, 16))
        fg, affs = np.array(outputs[0, 0]), np.array(outputs[0, 1:])
        mws = segmentation.mutex_watershed_segmentation(fg, affs, common.OFFSETS, 100)

    else:  # inference using foreground-boundary inputs - for the unetr training
        outputs = predict_with_halo(input_, model, [device], block_shape=[384, 384], halo=[64, 64], disable_tqdm=True)
        fg, bd = outputs[0, :, :], outputs[1, :, :]
        ws1 = segmentation.watershed_from_components(bd, fg, min_size=10)
        ws2 = segmentation.watershed_from_maxima(bd, fg, min_size=10, min_distance=1)

    fname = os.path.split(img_path)[-1]
    with h5py.File(os.path.join(root_save_dir, f"src-{ctype}", f"{fname}.h5"), "a") as f:
        ds = f.require_dataset("foreground", shape=fg.shape, compression="gzip", dtype=fg.dtype)
        ds[:] = fg
        if with_affinities:
            ds = f.require_dataset("affinities", shape=affs.shape, compression="gzip", dtype=affs.dtype)
            ds[:] = affs
            ds = f.require_dataset("segmentation", shape=mws.shape, compression="gzip", dtype=mws.dtype)
            ds[:] = mws
        else:
            ds = f.require_dataset("boundary", shape=bd.shape, compression="gzip", dtype=bd.dtype)
            ds[:] = bd
            ds = f.require_dataset("watershed1", shape=ws1.shape, compression="gzip", dtype=ws1.dtype)
            ds[:] = ws1
            ds = f.require_dataset("watershed2", shape=ws2.shape, compression="gzip", dtype=ws2.dtype)
            ds[:] = ws2


#
# miscellanous utilities
#


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", action='store_true', help="Enables UNETR training on LiveCell dataset"
    )

    parser.add_argument(
        "--predict", action='store_true', help="Enables UNETR prediction on LiveCell dataset"
    )

    parser.add_argument(
        "--evaluate", action='store_true', help="Enables UNETR evaluation on LiveCell dataset"
    )

    parser.add_argument(
        "--source_choice", type=str, default="torch-em",
        help="The source where the model comes from, i.e. either torch-em / monai"
    )

    parser.add_argument(
        "-m", "--model_name", type=str, default="vit_b", help="Name of the ViT to use as the encoder in UNETR"
    )

    parser.add_argument(
        "--do_sam_ini", action='store_true', help="Enables initializing UNETR with SAM's ViT weights"
    )

    parser.add_argument(
        "-c", "--cell_type", type=str, default=None, help="Choice of cell-type for doing the training"
    )

    parser.add_argument(
        "-i", "--input", type=str, default="/scratch/usr/nimanwai/data/livecell",
        help="Path where the dataset already exists/will be downloaded by the dataloader"
    )

    parser.add_argument(
        "-s", "--save_root", type=str, default="/scratch/usr/nimanwai/models/unetr/",
        help="Path where checkpoints and logs will be saved"
    )

    parser.add_argument(
        "--save_dir", type=str, default="/scratch/usr/nimanwai/predictions/unetr",
        help="Path to save predictions from UNETR model"
    )

    parser.add_argument(
        "--with_affinities", action="store_true",
        help="Trains the UNETR model with affinities"
    )

    parser.add_argument("--iterations", type=int, default=100000)
    return parser


def get_loss_function(with_affinities=True):
    if with_affinities:
        loss = LossWrapper(
            loss=DiceLoss(),
            transform=ApplyAndRemoveMask(masking_method="multiply")
        )
    else:
        loss = DiceLoss()

    return loss
