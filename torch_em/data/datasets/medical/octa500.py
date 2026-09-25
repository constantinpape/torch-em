"""The OCTA-500 dataset contains annotations for retinal vessel segmentation (large vessels, capillaries,
arteries, veins) and foveal avascular zone (FAZ) segmentation in en-face projections derived from 3D
OCT / OCTA volumes.

The dataset comprises 500 subjects, split into two field-of-view (FOV) subsets: 'OCTA_6M' (subject ids
10001-10300, FOV 6mm x 6mm x 2mm, volume shape 400 x 400 x 640) and 'OCTA_3M' (subject ids 10301-10500,
FOV 3mm x 3mm x 2mm, volume shape 304 x 304 x 640). For each subject, six 2D en-face projection maps are
derived from the 3D OCT / OCTA volumes: 'OCT(FULL)', 'OCT(ILM_OPL)', 'OCT(OPL_BM)', 'OCTA(FULL)',
'OCTA(ILM_OPL)' and 'OCTA(OPL_BM)'. Pixel-wise segmentation masks are provided for large vessels,
capillaries, arteries, veins and the FAZ, matching the resolution of the projection maps.

NOTE: The dataset also provides 2D / 3D FAZ and retinal layer annotations in the original release, but
this loader only covers the vessel and FAZ label types listed in `LABEL_DIRS` above, whose folder layout
and file format could be corroborated from the dataset's accompanying publications and downstream usage.
The retinal layer annotations are not supported here, as their exact on-disk format could not be verified
without direct access to the (gated) data.

NOTE: This dataset is hosted on IEEE DataPort at https://ieee-dataport.org/open-access/octa-500 and is
gated: downloading it requires a free IEEE account (or IEEE Society membership) to view the page, and
the password-protected archives themselves require directly emailing the dataset's authors. Automatic
download is not supported, see `get_octa500_data` for the exact manual steps.

The dataset is from the publication https://doi.org/10.1016/j.media.2024.103092 (and the earlier preprint
https://doi.org/10.48550/arXiv.2012.07261). Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


SUBSET_IDS = {
    "6M": range(10001, 10301),
    "3M": range(10301, 10501),
}
"""Mapping from the subset choice to its range of subject ids."""

PROJECTIONS = {
    "oct_full": "OCT(FULL)",
    "oct_ilm_opl": "OCT(ILM_OPL)",
    "oct_opl_bm": "OCT(OPL_BM)",
    "octa_full": "OCTA(FULL)",
    "octa_ilm_opl": "OCTA(ILM_OPL)",
    "octa_opl_bm": "OCTA(OPL_BM)",
}
"""Mapping from the projection choice to its folder in the released data."""

LABEL_DIRS = {
    "large_vessel": "GT_LargeVessel",
    "capillary": "GT_Capillary",
    "artery": "GT_Artery",
    "vein": "GT_Vein",
    "faz": "GT_FAZ",
}
"""Mapping from the label type choice to its folder in the released data."""


def get_octa500_data(
    path: Union[os.PathLike, str], subset: Literal["3M", "6M"], download: bool = False
) -> str:
    """Obtain the OCTA-500 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        subset: The choice of field-of-view subset. Either '3M' or '6M'.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder where the subset data is expected to be stored.
    """
    if subset not in SUBSET_IDS:
        raise ValueError(f"'{subset}' is not a valid subset. Choose from {list(SUBSET_IDS.keys())}.")

    data_dir = os.path.join(path, f"OCTA_{subset}")
    if os.path.exists(data_dir):
        return data_dir

    if download:
        msg = "Download is set to True, but 'torch_em' cannot download this dataset automatically."
        raise NotImplementedError(msg)

    raise RuntimeError(
        "The OCTA-500 dataset is hosted on IEEE DataPort and cannot be downloaded automatically. "
        "Please follow these steps to obtain it manually:\n"
        "1. Visit https://ieee-dataport.org/open-access/octa-500 and sign in with a free IEEE account "
        "(or IEEE Society membership), which is required to view the download page.\n"
        "2. The archives are password-protected. Send an email to chen2qiang@njust.edu.cn with the "
        "subject line 'OCTA500: [your organization]: [your name]' to request the password.\n"
        "3. Once you have the password, download 'Label.zip' and the 'OCTA_3mm_part*.zip' / "
        "'OCTA_6mm_part*.zip' archives from the IEEE DataPort page and extract them.\n"
        f"4. Place (or symlink) the extracted '3M' subset at '{os.path.join(path, 'OCTA_3M')}' and the "
        f"'6M' subset at '{os.path.join(path, 'OCTA_6M')}', each containing a 'Projection Maps' folder "
        "and the 'GT_LargeVessel' / 'GT_Capillary' / 'GT_Artery' / 'GT_Vein' / 'GT_FAZ' label folders.\n"
        f"Expected location for the requested '{subset}' subset: '{data_dir}'."
    )


def get_octa500_paths(
    path: Union[os.PathLike, str],
    subset: Literal["3M", "6M"],
    label_type: Literal["large_vessel", "capillary", "artery", "vein", "faz"] = "large_vessel",
    projection: Literal[
        "oct_full", "oct_ilm_opl", "oct_opl_bm", "octa_full", "octa_ilm_opl", "octa_opl_bm"
    ] = "octa_full",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the OCTA-500 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        subset: The choice of field-of-view subset. Either '3M' or '6M'.
        label_type: The choice of segmentation label. One of 'large_vessel', 'capillary', 'artery',
            'vein' or 'faz'.
        projection: The choice of 2D en-face projection map used as the raw input.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if label_type not in LABEL_DIRS:
        raise ValueError(f"'{label_type}' is not a valid label type. Choose from {list(LABEL_DIRS.keys())}.")

    if projection not in PROJECTIONS:
        raise ValueError(f"'{projection}' is not a valid projection. Choose from {list(PROJECTIONS.keys())}.")

    data_dir = get_octa500_data(path, subset, download)

    image_dir = os.path.join(data_dir, "Projection Maps", PROJECTIONS[projection])
    label_dir = os.path.join(data_dir, LABEL_DIRS[label_type])

    image_paths, label_paths = [], []
    for subject_id in SUBSET_IDS[subset]:
        label_matches = glob(os.path.join(label_dir, f"{subject_id}.*"))
        if not label_matches:
            continue

        image_matches = glob(os.path.join(image_dir, f"{subject_id}.*"))
        if not image_matches:
            continue

        image_paths.append(image_matches[0])
        label_paths.append(label_matches[0])

    image_paths, label_paths = natsorted(image_paths), natsorted(label_paths)
    assert len(image_paths) == len(label_paths) and len(image_paths) > 0, \
        f"Could not find matching image and '{label_type}' label pairs for the '{subset}' subset in '{data_dir}'."

    return image_paths, label_paths


def get_octa500_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    subset: Literal["3M", "6M"],
    label_type: Literal["large_vessel", "capillary", "artery", "vein", "faz"] = "large_vessel",
    projection: Literal[
        "oct_full", "oct_ilm_opl", "oct_opl_bm", "octa_full", "octa_ilm_opl", "octa_opl_bm"
    ] = "octa_full",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the OCTA-500 dataset for retinal vessel and FAZ segmentation in OCTA en-face projections.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        subset: The choice of field-of-view subset. Either '3M' or '6M'.
        label_type: The choice of segmentation label. One of 'large_vessel', 'capillary', 'artery',
            'vein' or 'faz'.
        projection: The choice of 2D en-face projection map used as the raw input.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_octa500_paths(path, subset, label_type, projection, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_octa500_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    subset: Literal["3M", "6M"],
    label_type: Literal["large_vessel", "capillary", "artery", "vein", "faz"] = "large_vessel",
    projection: Literal[
        "oct_full", "oct_ilm_opl", "oct_opl_bm", "octa_full", "octa_ilm_opl", "octa_opl_bm"
    ] = "octa_full",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the OCTA-500 dataloader for retinal vessel and FAZ segmentation in OCTA en-face projections.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        subset: The choice of field-of-view subset. Either '3M' or '6M'.
        label_type: The choice of segmentation label. One of 'large_vessel', 'capillary', 'artery',
            'vein' or 'faz'.
        projection: The choice of 2D en-face projection map used as the raw input.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_octa500_dataset(
        path, patch_shape, subset, label_type, projection, resize_inputs, download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
