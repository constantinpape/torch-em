import os
import h5py
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

import vigra
import imageio.v3 as imageio
from skimage.segmentation import find_boundaries, watershed
from skimage.measure import label
from skimage.filters import sobel_h, sobel_v, gaussian
from skimage.morphology import disk, opening
from scipy.ndimage import binary_fill_holes

from elf.evaluation import dice_score, mean_segmentation_accuracy

import torch
import torch.nn as nn
import torch_em
from torch_em.util import segmentation
from torch_em.transform.raw import standardize
from torch_em.data.datasets import get_livecell_loader
from torch_em.loss import LossWrapper, ApplyAndRemoveMask
from torch_em.loss.dice import DiceLoss, DiceLossWithLogits
from torch_em.util.prediction import predict_with_halo, predict_with_padding
from torch_em.transform.raw import normalize

from micro_sam.util import get_centers_and_bounding_boxes


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


def get_distance_maps(labels):
    labels = vigra.analysis.relabelConsecutive(labels.astype("uint32"))[0]
    # compute the eccentricty centers
    # First expensive step: center computation (leave it here, it's done once per image)
    centers = vigra.filters.eccentricityCenters(labels.astype("uint32"))

    # 1 for all eccentricity centers of the cells 0 elsewhere
    center_mask = np.zeros_like(labels)
    for center in centers:
        center_mask[center] = 1

    x_distances = np.zeros(labels.shape, dtype="float32")
    y_distances = np.zeros(labels.shape, dtype="float32")

    _, bbox_coordinates = get_centers_and_bounding_boxes(labels, mode="p")

    def compute_distance_map(cell_id):
        mask = labels == cell_id

        # getting the bounding box coordinates for masking the roi
        bbox = bbox_coordinates[cell_id]

        # crop the respective inputs to the bbox shape (for getting the distance transforms in the roi)
        cropped_center_mask = center_mask[
            max(bbox[0], 0): min(bbox[2], mask.shape[-2]),
            max(bbox[1], 0): min(bbox[3], mask.shape[-1])
        ]
        cropped_mask = mask[
            max(bbox[0], 0): min(bbox[2], mask.shape[-2]),
            max(bbox[1], 0): min(bbox[3], mask.shape[-1])
        ]

        # compute directed distance to the current center
        # Second expensive step: compute the distance transform
        # this is done for each instance so we want to reduce the effort by restricting this to the bounding box
        this_distances = vigra.filters.vectorDistanceTransform(cropped_center_mask).transpose((2, 0, 1))  # directed distance transform applied to the centers
        this_y_distances, this_x_distances = this_distances[0], this_distances[1]

        # masking the distance transforms in the instances
        this_y_distances[~cropped_mask] = 0
        this_x_distances[~cropped_mask] = 0

        # nornmalize the distances
        this_y_distances /= np.abs(this_y_distances).max() + 1e-7
        this_x_distances /= np.abs(this_x_distances).max() + 1e-7

        # checks for making sure that our range is between [-1, 1] for both distance maps
        if np.abs(this_y_distances).max() > 1:
            raise RuntimeError(np.unique(this_y_distances))

        if np.abs(this_x_distances).max() > 1:
            raise RuntimeError(np.unique(this_x_distances))

        # set all distances outside of cells to 0
        y_distances[
            max(bbox[0], 0): min(bbox[2], mask.shape[-2]),
            max(bbox[1], 0): min(bbox[3], mask.shape[-1])
        ][cropped_mask] = this_y_distances[cropped_mask]
        x_distances[
            max(bbox[0], 0): min(bbox[2], mask.shape[-2]),
            max(bbox[1], 0): min(bbox[3], mask.shape[-1])
        ][cropped_mask] = this_x_distances[cropped_mask]

    cell_ids = np.unique(labels)[1:]  # excluding background id
    for cell_id in cell_ids:
        compute_distance_map(cell_id)

    binary_labels = labels > 0
    return np.stack([binary_labels, y_distances, x_distances], axis=0)  # channels - 0:binary, 1:vertical, 2:horizontal


def get_my_livecell_loaders(
        input_path: str,
        patch_shape: Tuple[int, int],
        cell_types: Optional[str] = None,
        with_binary: bool = False,
        with_boundary: bool = False,
        with_affinities: bool = False,
        with_distance_maps: bool = False,
):
    """Returns the LIVECell training and validation dataloaders
    """

    if with_distance_maps:
        label_trafo = torch_em.transform.label.PerObjectDistances(
            distances=True,
            boundary_distances=True,
            directed_distances=False,
            foreground=True,
            min_size=10,
        )
    else:
        label_trafo = None

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
        boundaries=with_boundary,  # this returns dataloaders with foreground and boundary channels
        binary=with_binary,
        label_transform=label_trafo,
        label_dtype=torch.float32
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
        boundaries=with_boundary,  # this returns dataloaders with foreground and boundary channels
        binary=with_binary,
        label_transform=label_trafo,
        label_dtype=torch.float32
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
        raise ValueError(f"The available UNETR models are either from \"torch-em\" or \"monai\", choose from them instead of - {source_choice}")

    return model


#
# LIVECELL UNETR INFERENCE - foreground boundary / foreground affinities / foreground dist. maps
#


def hovernet_instance_segmentation(prediction, threshold1=0.5, threshold2=0.4, min_size=250):
    """Adapted from HoVerNet's post-processing:
        - https://github.com/vqdang/hover_net/blob/master/models/hovernet/post_proc.py
    """
    # let's get the channel-wise components for separate use-cases
    binary_raw = prediction[0, ...]
    v_map_raw = prediction[1, ...]
    h_map_raw = prediction[2, ...]

    # connected components
    cc = label((binary_raw >= threshold1).astype(np.int32))
    cc = segmentation.size_filter(cc, min_size)
    cc[cc > 0] = 1

    # sobel filter on the respective (normalized) distance maps
    # NOTE: the convention here follows our logic of implementation,
    # hence has been reversed
    # EXPLANATION: we call distances along y axis as vertical maps,
    # and along x as horizontal
    # hence, we just pass the distance maps to the correct sobel filters
    # (see docstring of `sobel_v` and `sobel_h` for the filters used)
    v_grad = sobel_v(normalize(h_map_raw))
    h_grad = sobel_h(normalize(v_map_raw))

    v_grad = 1 - normalize(v_grad)
    h_grad = 1 - normalize(h_grad)

    overall = np.maximum(v_grad, h_grad)
    overall = overall - (1 - cc)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * cc
    # foreground values form mountains so inverse to get basins
    dist = -gaussian(dist)

    overall = np.array(overall >= threshold2, dtype=np.int32)

    marker = cc - overall
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker)  # Q: no idea why we do this twice in the org. impl

    # opening to get rid of small objects
    struc_element = disk(5)
    marker = opening(marker, struc_element)

    marker = label(marker)
    marker = segmentation.size_filter(marker, min_size)

    # watershed segmentation
    instances = watershed(dist, markers=marker, mask=cc)

    return instances


def predict_for_unetr(
        img_path, model, root_save_dir, device, with_affinities=False, with_distance_maps=False, ctype=None
):
    input_ = imageio.imread(img_path)
    input_ = standardize(input_)

    if with_affinities:  # inference using affinities
        outputs = predict_with_padding(model, input_, device=device, min_divisible=(16, 16))
        fg, affs = np.array(outputs[0, 0]), np.array(outputs[0, 1:])
        mws = segmentation.mutex_watershed_segmentation(fg, affs, offsets=OFFSETS)

    elif with_distance_maps:  # inference using foreground and hv distance maps
        outputs = predict_with_padding(model, input_, device=device, min_divisible=(16, 16))
        fg, cdist, bdist = outputs.squeeze()
        dm_seg = segmentation.watershed_from_center_and_boundary_distances(cdist, bdist, fg, min_size=50)

    else:  # inference using foreground-boundary inputs - for the unetr training
        outputs = predict_with_halo(input_, model, [device], block_shape=[384, 384], halo=[64, 64], disable_tqdm=True)
        fg, bd = outputs[0, :, :], outputs[1, :, :]
        ws1 = segmentation.watershed_from_components(bd, fg)
        ws2 = segmentation.watershed_from_maxima(bd, fg, min_distance=1)

    fname = Path(img_path).stem
    save_path = os.path.join(root_save_dir, "src-all" if ctype is None else f"src-{ctype}", f"{fname}.h5")
    with h5py.File(save_path, "a") as f:
        if with_affinities:
            ds = f.require_dataset("foreground", shape=fg.shape, compression="gzip", dtype=fg.dtype)
            ds[:] = fg
            ds = f.require_dataset("affinities", shape=affs.shape, compression="gzip", dtype=affs.dtype)
            ds[:] = affs
            ds = f.require_dataset("segmentation", shape=mws.shape, compression="gzip", dtype=mws.dtype)
            ds[:] = mws

        elif with_distance_maps:
            ds = f.require_dataset("segmentation", shape=dm_seg.shape, compression="gzip", dtype=dm_seg.dtype)
            ds[:] = dm_seg

        else:
            ds = f.require_dataset("foreground", shape=fg.shape, compression="gzip", dtype=fg.dtype)
            ds[:] = fg
            ds = f.require_dataset("boundary", shape=bd.shape, compression="gzip", dtype=bd.dtype)
            ds[:] = bd
            ds = f.require_dataset("watershed1", shape=ws1.shape, compression="gzip", dtype=ws1.dtype)
            ds[:] = ws1
            ds = f.require_dataset("watershed2", shape=ws2.shape, compression="gzip", dtype=ws2.dtype)
            ds[:] = ws2


#
# LIVECELL UNETR EVALUATION - foreground boundary / foreground affinities
#

def evaluate_for_unetr(gt_path, _save_dir, with_affinities=False, with_distance_maps=False):
    fname = Path(gt_path).stem
    gt = imageio.imread(gt_path)

    output_file = os.path.join(_save_dir, f"{fname}.h5")
    with h5py.File(output_file, "r") as f:
        if with_affinities:
            mws = f["segmentation"][:]

        elif with_distance_maps:
            instances = f["segmentation"][:]

        else:
            fg = f["foreground"][:]
            bd = f["boundary"][:]
            ws1 = f["watershed1"][:]
            ws2 = f["watershed2"][:]

    if with_affinities:
        mws_msa, mws_sa_acc = mean_segmentation_accuracy(mws, gt, return_accuracies=True)
        return mws_msa, mws_sa_acc[0]

    elif with_distance_maps:
        instances_msa, instances_sa_acc = mean_segmentation_accuracy(instances, gt, return_accuracies=True)
        return instances_msa, instances_sa_acc[0]

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

    # this argument takes care of which ViT encoder to use for the UNETR (as ViTs from SAM and MAE are different)
    parser.add_argument(
        "--pretrained_choice", type=str, default="sam",
    )

    parser.add_argument(
        "--with_affinities", action="store_true",
        help="Trains the UNETR model with affinities"
    )

    parser.add_argument("--iterations", type=int, default=100000)
    return parser


#
# LOSS FUNCTIONS
#


class HoVerNetLoss(nn.Module):
    """Computes the overall loss for the HoVer-Net style training
    Reference: https://github.com/vqdang/hover_net/blob/master/models/hovernet/utils.py

    Arguments:
        compute_dice: The function to compute the dice loss (default: None)
            - If `None`, we use the implementation from `torch_em.loss`
        compute_mse: The function to compute the mse loss (default: None)
            - If `None`, we use the implementation from PyTorch.
        compute_bce: The function to compute the binary cross entropy loss (default: None)
            - If `None`, we use the implementation from PyTorch.
        device: To move the respective tensors to desired device (default: None)
            - If `None`, we make use of `cuda` if GPU is found, else use `cpu` instead
    """
    def __init__(
            self,
            compute_dice=None,
            compute_mse=None,
            compute_bce=None,
            device=None,
            sobel_kernel_size: int = 5
    ):
        super().__init__()
        self.compute_dice = DiceLossWithLogits() if compute_dice is None else compute_dice
        self.compute_mse = nn.MSELoss() if compute_mse is None else compute_mse
        self.compute_bce = nn.BCEWithLogitsLoss() if compute_bce is None else compute_bce

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.sobel_kernel_size = sobel_kernel_size

    def get_sobel_kernel(self, size):
        "Get the sobel kernel of a given window size"
        assert size % 2 == 1, f"The expected window size should be odd, but {size} was passed"
        hrange = torch.arange(-size // 2+1, size // 2+1, dtype=torch.float32)
        vrange = torch.arange(-size // 2+1, size // 2+1, dtype=torch.float32)
        h, v = torch.meshgrid(hrange, vrange, indexing="xy")
        kernel_v = v / (h*h + v*v + 1e-15)
        kernel_h = h / (h*h + v*v + 1e-15)
        return kernel_v, kernel_h

    def get_distance_gradients(self, v_map, h_map):
        "Calculates the gradients of the respective distance maps"
        kernel_v, kernel_h = self.get_sobel_kernel(self.sobel_kernel_size)
        kernel_v = kernel_v.view(1, 1, self.sobel_kernel_size, self.sobel_kernel_size).to(self.device)
        kernel_h = kernel_h.view(1, 1, self.sobel_kernel_size, self.sobel_kernel_size).to(self.device)

        v_ch = v_map[:, None, ...]
        h_ch = h_map[:, None, ...]

        g_v = nn.functional.conv2d(v_ch, kernel_v, padding=2)
        g_h = nn.functional.conv2d(h_ch, kernel_h, padding=2)
        return torch.cat([g_v, g_h], dim=1)

    def compute_msge(self, input_, target, focus):
        "Computes the mse loss for the respective gradients of distance maps and combines them together"
        input_vmap, input_hmap = input_[:, 0, ...], input_[:, 1, ...]
        target_vmap, target_hmap = target[:, 0, ...], target[:, 1, ...]

        input_grad = self.get_distance_gradients(input_vmap, input_hmap)
        target_grad = self.get_distance_gradients(target_vmap, target_hmap)
        msge_loss = self.compute_mse(input_grad * focus, target_grad * focus)
        return msge_loss

    def get_np_branch_loss(self, input_, target):
        "Computes the loss for the binary predictions w.r.t. the ground truth."
        input_, target = input_[:, None, ...], target[:, None, ...]
        dice_loss = self.compute_dice(input_, target)
        bce_loss = self.compute_bce(input_, target)
        return dice_loss, bce_loss

    def get_hv_branch_loss(self, input_, target, focus):
        "Computes the loss for the distances maps w.r.t. their respective ground truth."
        focus = torch.cat([focus[:, None, ...]] * target.shape[0], dim=1)

        # mean squared error loss of combined predicted hv distance maps w.r.t. the true hv distance maps
        mse_loss = self.compute_mse(input_, target)

        # mean squared error loss of the gradients of predicted v & h distance maps w.r.t. the true v & h maps
        msge_loss = self.compute_msge(input_, target, focus)

        return mse_loss, msge_loss

    def forward(self, input_, target):
        # expected shape of both `input_` and `target` is (B*3*H*W)
        # first channel is binary predictions; secound and third channels are vertical and horizontal maps respectively
        assert input_.shape == target.shape, input_.shape

        fg_input_, fg_target = input_[:, 0, ...], target[:, 0, ...]
        dice_loss, bce_loss = self.get_np_branch_loss(fg_input_, fg_target)

        vh_input_, vh_target = input_[:, 1:, ...], target[:, 1:, ...]
        mse_loss, msge_loss = self.get_hv_branch_loss(vh_input_, vh_target, focus=fg_target)

        # losses added together to get overall loss
        #     - for foreground background channel: losses added together to get overall loss (1 * (BCE + DICE))
        #     - for distance maps: 1 * MSE + 2 * MSGE - HoVerNet's empirical selection)
        overall_loss = dice_loss + bce_loss + mse_loss + 2 * msge_loss
        return overall_loss


class DistanceLoss(nn.Module):
    def __init__(self, mask_distances_in_bg):
        super().__init__()

        self.dice_loss = DiceLoss()
        self.mse_loss = nn.MSELoss()
        self.mask_distances_in_bg = mask_distances_in_bg

    def forward(self, input_, target):
        assert input_.shape == target.shape, input_.shape
        assert input_.shape[1] == 3, input_.shape

        fg_input, fg_target = input_[:, 0, ...], target[:, 0, ...]
        fg_loss = self.dice_loss(fg_target, fg_input)

        cdist_input, cdist_target = input_[:, 1, ...], target[:, 1, ...]
        if self.mask_distances_in_bg:
            cdist_loss = self.mse_loss(cdist_target * fg_target, cdist_input * fg_target)
        else:
            cdist_loss = self.mse_loss(cdist_target, cdist_input)

        bdist_input, bdist_target = input_[:, 2, ...], target[:, 2, ...]
        if self.mask_distances_in_bg:
            bdist_loss = self.mse_loss(bdist_target * fg_target, bdist_input * fg_target)
        else:
            bdist_loss = self.mse_loss(bdist_target, bdist_input)

        overall_loss = fg_loss + cdist_loss + bdist_loss
        return overall_loss


def get_loss_function(with_affinities=False, with_distance_maps=False):
    if with_affinities:
        loss = LossWrapper(
            loss=DiceLoss(),
            transform=ApplyAndRemoveMask(masking_method="multiply")
        )
    elif with_distance_maps:
        # loss = HoVerNetLoss()
        # Updated the loss function for the simplfied distance loss.
        # TODO we can try both with and without masking
        loss = DistanceLoss(mask_distances_in_bg=True)
    else:
        loss = DiceLoss()

    return loss


#
# HOVERNET ORIGINAL IMPLEMENTATIONS
#


def opencv_hovernet_instance_segmentation(pred):
    "https://github.com/vqdang/hover_net/blob/master/models/hovernet/post_proc.py"
    import cv2
    from scipy.ndimage import measurements
    from scipy import ndimage

    def remove_small_objects(pred, min_size=64, connectivity=1):
        out = pred

        if min_size == 0:  # shortcut for efficiency
            return out

        if out.dtype == bool:
            selem = ndimage.generate_binary_structure(pred.ndim, connectivity)
            ccs = np.zeros_like(pred, dtype=np.int32)
            ndimage.label(pred, selem, output=ccs)
        else:
            ccs = out

        try:
            component_sizes = np.bincount(ccs.ravel())
        except ValueError:
            raise ValueError(
                "Negative value labels are not supported. Try "
                "relabeling the input with `scipy.ndimage.label` or "
                "`skimage.morphology.label`."
            )

        too_small = component_sizes < min_size
        too_small_mask = too_small[ccs]
        out[too_small_mask] = 0

        return out

    pred = pred.transpose(1, 2, 0)  # making channels last

    pred = np.array(pred, dtype=np.float32)

    blb_raw = pred[..., 0]
    h_dir_raw = pred[..., 2]
    v_dir_raw = pred[..., 1]

    # processing
    blb = np.array(blb_raw >= 0.5, dtype=np.int32)

    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1  # background is 0 already

    h_dir = cv2.normalize(h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    v_dir = cv2.normalize(v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)

    sobelh = 1 - (cv2.normalize(sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
    sobelv = 1 - (cv2.normalize(sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb
    # nuclei values form mountains so inverse to get basins
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    marker = blb - overall
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = measurements.label(marker)[0]
    marker = remove_small_objects(marker, min_size=10)

    proced_pred = watershed(dist, markers=marker, mask=blb)

    return proced_pred


def gen_instance_hv_map(ann, crop_shape=(512, 512)):
    "https://github.com/vqdang/hover_net/blob/master/models/hovernet/targets.py"
    from scipy.ndimage import measurements
    from skimage import morphology as morph

    def fix_mirror_padding(ann):
        current_max_id = np.amax(ann)
        inst_list = list(np.unique(ann))
        inst_list.remove(0)  # 0 is background
        for inst_id in inst_list:
            inst_map = np.array(ann == inst_id, np.uint8)
            remapped_ids = measurements.label(inst_map)[0]
            remapped_ids[remapped_ids > 1] += current_max_id
            ann[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
            current_max_id = np.amax(ann)
        return ann

    def cropping_center(x, crop_shape, batch=False):
        orig_shape = x.shape
        if not batch:
            h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
            w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
            x = x[h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
        else:
            h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
            w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
            x = x[:, h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
        return x

    def get_bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        # due to python indexing, need to add 1 to max
        # else accessing will be 1px in the box, not out
        rmax += 1
        cmax += 1
        return [rmin, rmax, cmin, cmax]

    orig_ann = ann.copy()  # instance ID map
    fixed_ann = fix_mirror_padding(orig_ann)
    # re-cropping with fixed instance id map
    crop_ann = cropping_center(fixed_ann, crop_shape)

    # TODO: deal with 1 label warning
    crop_ann = morph.remove_small_objects(crop_ann, min_size=30)

    x_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
    y_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)

    inst_list = list(np.unique(crop_ann))
    inst_list.remove(0)  # 0 is background
    for inst_id in inst_list:
        inst_map = np.array(fixed_ann == inst_id, np.uint8)
        inst_box = get_bounding_box(inst_map)

        # expand the box by 2px
        # Because we first pad the ann at line 207, the bboxes
        # will remain valid after expansion
        inst_box[0] -= 2
        inst_box[2] -= 2
        inst_box[1] += 2
        inst_box[3] += 2

        inst_map = inst_map[inst_box[0]: inst_box[1], inst_box[2]: inst_box[3]]

        if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
            continue

        # instance center of mass, rounded to nearest pixel
        inst_com = list(measurements.center_of_mass(inst_map))

        inst_com[0] = int(inst_com[0] + 0.5)
        inst_com[1] = int(inst_com[1] + 0.5)

        inst_x_range = np.arange(1, inst_map.shape[1] + 1)
        inst_y_range = np.arange(1, inst_map.shape[0] + 1)
        # shifting center of pixels grid to instance center of mass
        inst_x_range -= inst_com[1]
        inst_y_range -= inst_com[0]

        inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

        # remove coord outside of instance
        inst_x[inst_map == 0] = 0
        inst_y[inst_map == 0] = 0
        inst_x = inst_x.astype("float32")
        inst_y = inst_y.astype("float32")

        # normalize min into -1 scale
        if np.min(inst_x) < 0:
            inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
        if np.min(inst_y) < 0:
            inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
        # normalize max into +1 scale
        if np.max(inst_x) > 0:
            inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
        if np.max(inst_y) > 0:
            inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

        ####
        x_map_box = x_map[inst_box[0]: inst_box[1], inst_box[2]: inst_box[3]]
        x_map_box[inst_map > 0] = inst_x[inst_map > 0]

        y_map_box = y_map[inst_box[0]: inst_box[1], inst_box[2]: inst_box[3]]
        y_map_box[inst_map > 0] = inst_y[inst_map > 0]

    hv_map = np.dstack([(orig_ann > 0).astype(np.int32), x_map, y_map]).transpose(2, 0, 1)  # HACK to convert to our desired outputs
    return hv_map
