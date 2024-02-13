import os
import h5py
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Optional

import imageio.v3 as imageio
from skimage.segmentation import find_boundaries
from elf.evaluation import dice_score, mean_segmentation_accuracy

import torch
import torch_em
from torch_em.util import segmentation
from torch_em.transform.raw import standardize
from torch_em.data.datasets import get_livecell_loader
from torch_em.util.prediction import predict_with_halo, predict_with_padding
from torch_em.loss import DiceLoss, LossWrapper, ApplyAndRemoveMask, DiceBasedDistanceLoss

try:
    from micro_sam.training import identity
except ModuleNotFoundError:
    import warnings
    warnings.warn("`micro_sam` could not be imported, hence we build an identity fn for the raw transform.")

    def identity(raw):
        return raw


OFFSETS = [
    [-1, 0], [0, -1],
    [-3, 0], [0, -3],
    [-9, 0], [0, -9],
    [-27, 0], [0, -27]
]


CELL_TYPES = ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3"]


#
# LIVECELL TRAINING SCHEME
#


def do_unetr_training(
    train_loader,
    val_loader,
    model,
    loss,
    device=None,
    iterations=1e5,
    save_root=None,
    name="livecell-unetr",
    learning_rate=1e-5,
):
    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        device=device,
        mixed_precision=True,
        log_image_interval=50,
        compile_model=False,
        save_root=save_root,
        loss=loss,
        metric=loss
    )
    trainer.fit(iterations)


#
# LIVECELL DATALOADERS
#


def get_my_livecell_loaders(
    input_path: str,
    patch_shape: Tuple[int, int],
    experiment_name: str,
    cell_types: Optional[str] = None,
    input_norm: bool = True  # if True, use default raw trafo, else use identity raw trafo
):
    """Returns the LiveCELL training and validation dataloaders
    """

    if experiment_name == "distances":
        label_trafo = torch_em.transform.label.PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False, foreground=True, min_size=25
        )
    else:
        label_trafo = None

    if input_norm:
        print("Using default raw transform...")
        raw_transform = torch_em.transform.get_raw_transform()
    else:
        print("Using identity raw transform...")
        raw_transform = identity

    train_loader = get_livecell_loader(
        path=input_path,
        split="train",
        patch_shape=patch_shape,
        batch_size=2,
        download=True,
        num_workers=16,
        cell_types=None if cell_types is None else [cell_types],
        # this returns dataloaders with affinity channels and foreground-background channels
        offsets=OFFSETS if experiment_name == "affinities" else None,
        boundaries=(experiment_name == "boundaries"),  # this returns dataloaders with foreground and boundary channels
        label_transform=label_trafo,
        label_dtype=torch.float32,
        raw_transform=raw_transform
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
        offsets=OFFSETS if experiment_name == "affinities" else None,
        boundaries=(experiment_name == "boundaries"),  # this returns dataloaders with foreground and boundary channels
        label_transform=label_trafo,
        label_dtype=torch.float32,
        raw_transform=raw_transform
    )

    return train_loader, val_loader


#
# UNETR MODEL(S) FROM MONAI AND torch_em
#

MODELS = {
    "vit_b": "/scratch/projects/nim00007/sam/vanilla/sam_vit_b_01ec64.pth",
    "vit_h": "/scratch/projects/nim00007/sam/vanilla/sam_vit_h_4b8939.pth"
}


def get_unet_model(output_channels, use_conv_transpose):
    from torch_em.model.unet import UNet2d, Upsampler2d
    from torch_em.model.unetr import SingleDeconv2DBlock
    model = UNet2d(
        in_channels=1,
        out_channels=output_channels,
        initial_features=64,
        final_activation="Sigmoid",
        sampler_impl=SingleDeconv2DBlock if use_conv_transpose else Upsampler2d
    )
    return model


def get_unetr_model(
    model_name: str,
    source_choice: str,
    patch_shape: Tuple[int, int],
    sam_initialization: bool,
    output_channels: int,
    use_conv_transpose: bool,
    backbone: str = "sam",
):
    """Returns the expected UNETR model
    """
    if source_choice == "torch-em":
        # this returns the unetr model whihc uses the vision transformer from segment anything
        from torch_em import model as torch_em_models
        model = torch_em_models.UNETR(
            backbone=backbone,
            encoder=model_name,
            out_channels=output_channels,
            use_sam_stats=sam_initialization,
            final_activation="Sigmoid",
            encoder_checkpoint=MODELS[model_name] if sam_initialization else None,
            use_conv_transpose=use_conv_transpose
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
        model.out_channels = 2

    else:
        raise ValueError(f"The available UNETR models are either from `torch-em` or `monai`. \
                         Please choose from them instead of {source_choice}")

    return model


#
# LIVECELL UNETR INFERENCE - foreground boundary / foreground affinities / foreground dist. maps
#


def do_unetr_inference(
    input_path: str,
    device: torch.device,
    model,
    root_save_dir: str,
    save_root: str,
    experiment_name: str,
    name_extension: str,
    input_norm: bool = True,
):
    test_img_dir = os.path.join(input_path, "images", "livecell_test_images", "*")
    model_ckpt = os.path.join(save_root, "checkpoints", name_extension, "best.pt")
    assert os.path.exists(model_ckpt), model_ckpt

    model.load_state_dict(torch.load(model_ckpt, map_location=torch.device('cpu'))["model_state"])
    model.to(device)
    model.eval()

    # creating the respective directories for saving the outputs
    os.makedirs(os.path.join(root_save_dir), exist_ok=True)

    with torch.no_grad():
        for img_path in tqdm(glob(test_img_dir), desc=f"Run inference for all livecell with model {model_ckpt}"):
            predict_for_unetr(img_path, model, root_save_dir, device, experiment_name, input_norm)


def predict_for_unetr(
    img_path, model, root_save_dir, device, experiment_name, input_norm=True,
):
    fname = Path(img_path).stem
    save_path = os.path.join(root_save_dir, f"{fname}.h5")
    if os.path.exists(save_path):
        return

    input_ = imageio.imread(img_path)
    if input_norm:
        input_ = standardize(input_)

    if experiment_name == "affinities":  # inference using affinities
        outputs = predict_with_padding(model, input_, device=device, min_divisible=(16, 16))
        fg, affs = np.array(outputs[0, 0]), np.array(outputs[0, 1:])
        mws = segmentation.mutex_watershed_segmentation(fg, affs, offsets=OFFSETS)

    elif experiment_name == "distances":  # inference using foreground and hv distance maps
        outputs = predict_with_padding(model, input_, device=device, min_divisible=(16, 16))
        fg, cdist, bdist = outputs.squeeze()
        dm_seg = segmentation.watershed_from_center_and_boundary_distances(
            cdist, bdist, fg, min_size=50,
            center_distance_threshold=0.5,
            boundary_distance_threshold=0.6,
            distance_smoothing=1.0
        )

    elif experiment_name == "boundaries":  # inference using foreground-boundary inputs - for the unetr training
        outputs = predict_with_halo(
            input_, model, [device], block_shape=[384, 384], halo=[64, 64],
            disable_tqdm=True, preprocess=standardize if input_norm else None
        )
        fg, bd = outputs
        ws1 = segmentation.watershed_from_components(bd, fg)
        ws2 = segmentation.watershed_from_maxima(bd, fg, min_distance=1)

    with h5py.File(save_path, "a") as f:
        ds = f.require_dataset("foreground", shape=fg.shape, compression="gzip", dtype=fg.dtype)
        ds[:] = fg

        if experiment_name == "affinities":
            ds = f.require_dataset("affinities", shape=affs.shape, compression="gzip", dtype=affs.dtype)
            ds[:] = affs
            ds = f.require_dataset("segmentation", shape=mws.shape, compression="gzip", dtype=mws.dtype)
            ds[:] = mws

        elif experiment_name == "distances":
            ds = f.require_dataset("cdist", shape=cdist.shape, compression="gzip", dtype=cdist.dtype)
            ds[:] = cdist
            ds = f.require_dataset("bdist", shape=bdist.shape, compression="gzip", dtype=bdist.dtype)
            ds[:] = bdist
            ds = f.require_dataset("segmentation", shape=dm_seg.shape, compression="gzip", dtype=dm_seg.dtype)
            ds[:] = dm_seg

        elif experiment_name == "boundaries":
            ds = f.require_dataset("boundary", shape=bd.shape, compression="gzip", dtype=bd.dtype)
            ds[:] = bd
            ds = f.require_dataset("watershed1", shape=ws1.shape, compression="gzip", dtype=ws1.dtype)
            ds[:] = ws1
            ds = f.require_dataset("watershed2", shape=ws2.shape, compression="gzip", dtype=ws2.dtype)
            ds[:] = ws2


#
# LIVECELL UNETR EVALUATION - foreground boundary / foreground affinities
#


def do_unetr_evaluation(
    input_path: str,
    root_save_dir: str,
    csv_save_dir: str,
    experiment_name: str,
):
    _save_dir = os.path.join(root_save_dir)
    assert os.path.exists(_save_dir), _save_dir

    gt_dir = os.path.join(input_path, "annotations", "livecell_test_images", "*", "*")

    res_path = os.path.join(csv_save_dir, "livecell.csv")
    if os.path.exists(res_path):
        print(pd.read_csv(res_path))
        print(f"Results are already saved at {res_path}")
        return

    msa_list, sa50_list = [], []
    fg_list, bd_list, msa1_list, sa501_list, msa2_list, sa502_list = [], [], [], [], [], []
    for gt_path in tqdm(glob(gt_dir)):
        all_metrics = evaluate_for_unetr(
            gt_path, _save_dir, experiment_name=experiment_name
        )
        if experiment_name in ["affinities", "distances"]:
            msa, sa50 = all_metrics
            msa_list.append(msa)
            sa50_list.append(sa50)

        else:
            fg_dice, bd_dice, msa1, sa_acc1, msa2, sa_acc2 = all_metrics
            fg_list.append(fg_dice)
            bd_list.append(bd_dice)
            msa1_list.append(msa1)
            sa501_list.append(sa_acc1[0])
            msa2_list.append(msa2)
            sa502_list.append(sa_acc2[0])

    if experiment_name in ["affinities", "distances"]:
        res_dict = {
            "LiveCELL": "Metrics",
            "mSA": np.mean(msa_list),
            "SA50": np.mean(sa50_list)
        }

    else:
        res_dict = {
            "LiveCELL": "Metrics",
            "ws1_mSA": np.mean(msa1_list),
            "ws1_SA50": np.mean(sa501_list),
            "ws2_mSA": np.mean(msa2_list),
            "ws2_SA50": np.mean(sa502_list)
        }

    df = pd.DataFrame.from_dict([res_dict])
    df.to_csv(res_path)
    print(df)
    print(f"Results are saved at {res_path}")


def evaluate_for_unetr(gt_path, _save_dir, experiment_name):
    fname = Path(gt_path).stem
    gt = imageio.imread(gt_path)

    output_file = os.path.join(_save_dir, f"{fname}.h5")
    with h5py.File(output_file, "r") as f:
        if experiment_name == "affinities":
            mws = f["segmentation"][:]

        elif experiment_name == "distances":
            instances = f["segmentation"][:]

        elif experiment_name == "boundaries":
            fg = f["foreground"][:]
            bd = f["boundary"][:]
            ws1 = f["watershed1"][:]
            ws2 = f["watershed2"][:]

    if experiment_name == "affinities":
        mws_msa, mws_sa_acc = mean_segmentation_accuracy(mws, gt, return_accuracies=True)
        return mws_msa, mws_sa_acc[0]

    elif experiment_name == "distances":
        instances_msa, instances_sa_acc = mean_segmentation_accuracy(instances, gt, return_accuracies=True)
        return instances_msa, instances_sa_acc[0]

    elif experiment_name == "boundaries":
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


def get_output_channels(experiment_name):
    if experiment_name == "boundaries":
        out_channels = 2
    elif experiment_name == "affinities":
        out_channels = len(OFFSETS) + 1
    elif experiment_name == "distances":
        out_channels = 3

    return out_channels


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true', help="Enables UNETR training on LiveCELL dataset")
    parser.add_argument("--predict", action='store_true', help="Enables UNETR prediction on LiveCELL dataset")
    parser.add_argument("--evaluate", action='store_true', help="Enables UNETR prediction on LiveCELL dataset")
    parser.add_argument(
        "--experiment_name", type=str, required=True, help="Choose from boundaries / affinities / distances"
    )

    parser.add_argument(
        "--use_unet", action='store_true', help="Use UNet2d for training on LiveCELL"
    )

    parser.add_argument("--patch_shape", type=int, nargs="+", default=(520, 704))

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
        "-s", "--save_root", type=str, required=True, help="Path where checkpoints and logs will be saved"
    )

    parser.add_argument("--iterations", type=int, default=100000)

    # this argument takes care of which ViT encoder to use for the UNETR (as ViTs from SAM and MAE are different)
    parser.add_argument("--pretrained_choice", type=str, default="sam")
    parser.add_argument(
        "--use_bilinear", action="store_true", help="Use bilinear interpolation for upsampling."
    )
    return parser


def get_loss_function(experiment_name):
    if experiment_name == "affinities":
        loss = LossWrapper(
            loss=DiceLoss(),
            transform=ApplyAndRemoveMask(masking_method="multiply")
        )

    elif experiment_name == "distances":
        loss = DiceBasedDistanceLoss(mask_distances_in_bg=True)

    elif experiment_name == "boundaries":
        loss = DiceLoss()

    return loss
