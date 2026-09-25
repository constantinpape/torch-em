"""The MAMA-MIA dataset contains annotations for primary breast tumor segmentation in
dynamic contrast-enhanced (DCE) MRI.

The dataset pools 1506 pre-treatment breast DCE-MRI of four public collections of The Cancer Imaging Archive
(DUKE, ISPY1, ISPY2 and NACT), for which the primary tumor was segmented by sixteen breast radiologists. The
segmentation is a binary mask of the primary tumor (which includes the non-mass enhancement areas), see
`LABEL_IDS`. The collection a case belongs to is encoded in its name and can be selected with the 'cohort'
argument.

NOTE: The official release at https://www.synapse.org/Synapse:syn60868042 requires a Synapse (or Health-RI XNAT)
account, so it cannot be downloaded automatically. This module uses the open redistribution at
https://huggingface.co/datasets/YongchengYAO/MAMA-MIA-Lite (CC BY-NC 4.0), which contains all 1506 cases with
their expert segmentations, but only the first post-contrast DCE phase (the phase the masks were drawn on) of
each case. The other DCE phases and the preliminary automatic segmentations of the official release are not part
of this redistribution, so all masks provided here are the expert segmentations. The volumes of the
redistribution were reoriented to RAS+ and the masks were cast to uint16.

The volumes are stored as nifti files with the slice axis last, but they are loaded with the slice axis first
(torch_em reverses the nifti axis order), so 2d patches are extracted along the axial axis.

This dataset is from the publication https://doi.org/10.1038/s41597-025-04707-4.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Literal, Optional

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "part1": "https://huggingface.co/datasets/YongchengYAO/MAMA-MIA-Lite/resolve/main/data-part001.zip",
    "part2": "https://huggingface.co/datasets/YongchengYAO/MAMA-MIA-Lite/resolve/main/data-part002.zip",
}

CHECKSUMS = {
    "part1": "8cbd3d0165a446793b9f342ef28dc27be4a7a81e4933f69515b5287fe6b0b89c",
    "part2": "22c87ab82574279c235f7863e9d2de0e8baa031ca998278b6a0a30722dabffee",
}

LABEL_IDS = {"background": 0, "primary_tumor": 1}

COHORTS = ["DUKE", "ISPY1", "ISPY2", "NACT"]

N_VOLUMES = 1506


def get_mama_mia_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the MAMA-MIA dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is stored.
    """
    if len(glob(os.path.join(path, "Images", "*.nii.gz"))) == N_VOLUMES:
        return path

    os.makedirs(path, exist_ok=True)

    for part, url in URLS.items():
        zip_path = os.path.join(path, f"{part}.zip")
        util.download_source(path=zip_path, url=url, download=download, checksum=CHECKSUMS[part])
        util.unzip(zip_path=zip_path, dst=path)

    return path


def get_mama_mia_paths(
    path: Union[os.PathLike, str],
    cohort: Optional[Literal["DUKE", "ISPY1", "ISPY2", "NACT"]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the MAMA-MIA data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        cohort: The choice of source collection. By default all four collections are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if cohort is not None and cohort not in COHORTS:
        raise ValueError(f"'{cohort}' is not a valid cohort. Please choose one of {COHORTS}.")

    data_dir = get_mama_mia_data(path, download)

    prefix = "*" if cohort is None else f"{cohort}_*"
    raw_paths = natsorted(glob(os.path.join(data_dir, "Images", f"{prefix}.nii.gz")))
    label_paths = [os.path.join(data_dir, "Masks", os.path.basename(p)) for p in raw_paths]
    assert len(raw_paths) > 0 and all(os.path.exists(p) for p in label_paths)

    return raw_paths, label_paths


def get_mama_mia_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    cohort: Optional[Literal["DUKE", "ISPY1", "ISPY2", "NACT"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the MAMA-MIA dataset for breast tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        cohort: The choice of source collection. By default all four collections are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_mama_mia_paths(path, cohort, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_mama_mia_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    cohort: Optional[Literal["DUKE", "ISPY1", "ISPY2", "NACT"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the MAMA-MIA dataloader for breast tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        cohort: The choice of source collection. By default all four collections are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_mama_mia_dataset(path, patch_shape, cohort, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
