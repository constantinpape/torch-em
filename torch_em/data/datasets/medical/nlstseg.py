"""The NLSTseg dataset contains pixel-level annotations of lung cancer lesions in low-dose CT scans
of the National Lung Screening Trial (NLST).

The dataset consists of 605 patients with 715 manually annotated lesions (662 tumors and 53 nodules).
Each patient is stored as '<id>_CT.nii.gz' and '<id>_tumor.nii.gz', the patients are distributed over 6 zip archives
('2_LungTumor.zip' - '7_LungTumor.zip', about 34 GB in total). The labels are instance labels: 0 is the background
and each annotated lesion of a patient has its own id, starting from 1. Whether a lesion is a tumor or a nodule
is only recorded in the metadata table '1_Table.zip' ('Label.xlsx', column 'labels_type'), not in the masks.

The data is located at https://doi.org/10.5281/zenodo.14838349, released under a CC-BY-4.0 license.
NOTE: Older releases of this dataset are access restricted, this module uses the open release.

This dataset is from the publication https://doi.org/10.1038/s41597-025-05742-x.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Optional, Sequence

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL_BASE = "https://zenodo.org/records/14838349/files"

CHECKSUMS = {
    2: None,
    3: "1ecb954e525e61b6f1a618a68dd955c79066c568bfad4d6e50c77887cdf618d5",
    4: None,
    5: None,
    6: None,
    7: None,
}


def get_nlstseg_data(
    path: Union[os.PathLike, str], archives: Optional[Sequence[int]] = None, download: bool = False
) -> str:
    """Download the NLSTseg dataset.

    NOTE: The full dataset is about 34 GB. Use `archives` to only download a subset of it.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        archives: The numbers of the archives ('<number>_LungTumor.zip', from 2 to 7) to use.
            Each archive holds about 100 patients. By default all archives are used.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    archives = list(CHECKSUMS) if archives is None else list(archives)
    invalid = [n for n in archives if n not in CHECKSUMS]
    if invalid:
        raise ValueError(f"The archives {invalid} do not exist. Choose from {list(CHECKSUMS)}.")

    data_dir = os.path.join(path, "data")
    for number in archives:
        if os.path.exists(os.path.join(data_dir, f"NLSTseg_{number}_LungTumor")):
            continue

        os.makedirs(data_dir, exist_ok=True)
        zip_path = os.path.join(path, f"{number}_LungTumor.zip")
        util.download_source(
            path=zip_path, url=f"{URL_BASE}/{number}_LungTumor.zip?download=1", download=download,
            checksum=CHECKSUMS[number],
        )
        util.unzip(zip_path=zip_path, dst=data_dir)

    return data_dir


def get_nlstseg_paths(
    path: Union[os.PathLike, str], archives: Optional[Sequence[int]] = None, download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the NLSTseg data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        archives: The numbers of the archives ('<number>_LungTumor.zip', from 2 to 7) to use.
            By default all archives are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_nlstseg_data(path, archives, download)

    numbers = list(CHECKSUMS) if archives is None else list(archives)
    raw_paths = []
    for number in numbers:
        raw_paths.extend(glob(os.path.join(data_dir, f"NLSTseg_{number}_LungTumor", "*", "*_CT.nii.gz")))
    raw_paths = natsorted(raw_paths)
    label_paths = [p.replace("_CT.nii.gz", "_tumor.nii.gz") for p in raw_paths]

    assert len(raw_paths) > 0 and all(os.path.exists(p) for p in label_paths)

    return raw_paths, label_paths


def get_nlstseg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    archives: Optional[Sequence[int]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the NLSTseg dataset for lung lesion segmentation in low-dose CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        archives: The numbers of the archives ('<number>_LungTumor.zip', from 2 to 7) to use.
            By default all archives are used.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_nlstseg_paths(path, archives, download)

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
        is_seg_dataset=True,
        patch_shape=patch_shape,
        ndim=3,
        **kwargs
    )


def get_nlstseg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    archives: Optional[Sequence[int]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the NLSTseg dataloader for lung lesion segmentation in low-dose CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        archives: The numbers of the archives ('<number>_LungTumor.zip', from 2 to 7) to use.
            By default all archives are used.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_nlstseg_dataset(path, patch_shape, archives, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
