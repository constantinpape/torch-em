import os
import h5py
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

import imageio.v3 as imageio
from skimage.segmentation import find_boundaries
from elf.evaluation import dice_score, mean_segmentation_accuracy

from torch_em.util import segmentation
from torch_em.transform.raw import standardize
from torch_em.data.datasets import get_livecell_loader
from torch_em.loss import DiceLoss, LossWrapper, ApplyAndRemoveMask
from torch_em.util.prediction import predict_with_halo, predict_with_padding


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
        cell_types: Optional[str] = None,
        with_affinities: bool = False
):
    """Returns the LIVECell training and validation dataloaders
    """
    train_loader = get_livecell_loader(
        path=input_path,
        split="train",
        patch_shape=patch_shape,
        batch_size=2,
        download=True,
        num_workers=16,
        cell_types=None if cell_types is None else [cell_types],
        # this returns dataloaders with affinity channels and foreground-background channels
        offsets=OFFSETS if with_affinities else None,
        # this returns dataloaders with foreground and boundary channels
        boundaries=False if with_affinities else True
    )
    val_loader = get_livecell_loader(
        path=input_path,
        split="val",
        patch_shape=patch_shape,
        batch_size=1,
        download=True,
        num_workers=16,
        cell_types=None if cell_types is None else [cell_types],
        # this returns dataloaders with affinity channels and foreground-background channels
        offsets=OFFSETS if with_affinities else None,
        # this returns dataloaders with foreground and boundary channels
        boundaries=False if with_affinities else True
    )

    return train_loader, val_loader


#
# UNETR MODEL(S) FROM MONAI AND torch_em
#

MODELS = {
    "vit_b": "/scratch/projects/nim00007/sam/vanilla/sam_vit_b_01ec64.pth",
    "vit_h": "/scratch/projects/nim00007/sam/vanilla/sam_vit_h_4b8939.pth"
}


def get_unetr_model(
        model_name: str,
        source_choice: str,
        patch_shape: Tuple[int, int],
        sam_initialization: bool,
        output_channels: int,
        backbone: str = "sam"
):
    """Returns the expected UNETR model
    """
    if source_choice == "torch-em":
        # this returns the unetr model whihc uses the vision transformer from segment anything
        from torch_em import model as torch_em_models
        model = torch_em_models.UNETR(
            backbone=backbone, encoder=model_name, out_channels=output_channels,
            encoder_checkpoint_path=MODELS[model_name] if sam_initialization else None,
            use_sam_stats=sam_initialization  # FIXME: add mae weight initialization
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
        tmp_msg = "The available UNETR models are either from `torch-em` or `monai`. "
        tmp_msg += f"Please choose from them instead of {source_choice}"
        raise ValueError(tmp_msg)

    return model


#
# LIVECELL UNETR INFERENCE - foreground boundary / foreground affinities
#

def predict_for_unetr(img_path, model, root_save_dir, device, with_affinities, ctype=None):
    input_ = imageio.imread(img_path)
    input_ = standardize(input_)

    if with_affinities:  # inference using affinities
        outputs = predict_with_padding(model, input_, device=device, min_divisible=(16, 16))
        fg, affs = np.array(outputs[0, 0]), np.array(outputs[0, 1:])
        mws = segmentation.mutex_watershed_segmentation(fg, affs, offsets=OFFSETS)

    else:  # inference using foreground-boundary inputs - for the unetr training
        outputs = predict_with_halo(input_, model, [device], block_shape=[384, 384], halo=[64, 64], disable_tqdm=True)
        fg, bd = outputs[0, :, :], outputs[1, :, :]
        ws1 = segmentation.watershed_from_components(bd, fg)
        ws2 = segmentation.watershed_from_maxima(bd, fg, min_distance=1)

    fname = Path(img_path).stem
    save_path = os.path.join(root_save_dir, "src-all" if ctype is None else f"src-{ctype}", f"{fname}.h5")
    with h5py.File(save_path, "a") as f:
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
# LIVECELL UNETR EVALUATION - foreground boundary / foreground affinities
#

def evaluate_for_unetr(gt_path, _save_dir, with_affinities):
    fname = Path(gt_path).stem
    gt = imageio.imread(gt_path)

    output_file = os.path.join(_save_dir, f"{fname}.h5")
    with h5py.File(output_file, "r") as f:
        if with_affinities:
            mws = f["segmentation"][:]
        else:
            fg = f["foreground"][:]
            bd = f["boundary"][:]
            ws1 = f["watershed1"][:]
            ws2 = f["watershed2"][:]

    if with_affinities:
        mws_msa, mws_sa_acc = mean_segmentation_accuracy(mws, gt, return_accuracies=True)
        return mws_msa, mws_sa_acc[0]

    else:
        true_bd = find_boundaries(gt)

        # Compare the foreground prediction to the ground-truth.
        # Here, it's important not to threshold the segmentation. Otherwise EVERYTHING will be set to
        # foreground in the dice function, since we have a comparision > 0 in there, and everything in the
        # binary prediction evaluates to true.
        # For the GT we can set the threshold to 0, because this will map to the correct binary mask.
        fg_dice = dice_score(fg, gt, threshold_gt=0, threshold_seg=None)

        # Compare the background prediction to the ground-truth.
        # Here, we don't need any thresholds: for the prediction the same holds as before.
        # For the ground-truth we have already a binary label, so we don't need to threshold it again.
        bd_dice = dice_score(bd, true_bd, threshold_gt=None, threshold_seg=None)

        msa1, sa_acc1 = mean_segmentation_accuracy(ws1, gt, return_accuracies=True)  # type: ignore
        msa2, sa_acc2 = mean_segmentation_accuracy(ws2, gt, return_accuracies=True)  # type: ignore

        return fg_dice, bd_dice, msa1, sa_acc1, msa2, sa_acc2


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
