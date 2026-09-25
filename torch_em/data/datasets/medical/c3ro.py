"""The C3RO dataset contains crowdsourced radiotherapy contours on CT scans of five cancer sites: breast,
gastrointestinal (GI), gynecologic (GYN), head and neck (H&N) and sarcoma.

NOTE: This is NOT a large training set. The dataset has only five CT scans, one per cancer site. Its value is the
number of annotators: every scan was contoured by many independent raters, split into experts (4 to 15 per site)
and non-experts (26 to 124 per site), for 3 to 11 target structures per site (e.g. heart, parotid glands, GTV, CTV).
For every structure and annotator group a consensus mask, computed with STAPLE, is provided as well. This makes
the dataset useful for research on inter-rater variability and label uncertainty.

Each item of this loader pairs the CT scan of a site with the mask of a single structure. By default, this is the
STAPLE consensus of the expert raters. Use `annotator` to select the expert or non-expert group, and `rater` to
select the masks of a single rater of that group instead of the consensus. Individual raters additionally contoured
some auxiliary structures that have no consensus mask (e.g. 'body' or 'ptv'), which are then included as well.
Masks that do not match the shape of the CT are skipped (none of the 2477 NIfTI masks is affected).

The data is located at https://doi.org/10.6084/m9.figshare.21074182, released under a CC-BY-4.0 license.

Please cite the associated publication and the dataset if you use it for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List, Optional

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://ndownloader.figshare.com/files/42025569"
CHECKSUM = "74c94faa18ef2e78c512b1ec2a71755f9b93cba37d0aff19afeeb47a77f18887"

SITES = ["Breast", "GI", "GYN", "H&N", "Sarcoma"]
ANNOTATORS = {"expert": "Expert", "non_expert": "Non-Expert"}


def get_c3ro_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the C3RO dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "Organized_files_v4")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "Organized_files_v4.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path, remove=False)

    assert os.path.exists(data_dir), f"The extraction of the C3RO archive did not create '{data_dir}'."

    return data_dir


def get_c3ro_paths(
    path: Union[os.PathLike, str],
    site: Optional[Literal["Breast", "GI", "GYN", "H&N", "Sarcoma"]] = None,
    annotator: Literal["expert", "non_expert"] = "expert",
    rater: Optional[str] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the C3RO data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        site: The choice of cancer site. By default, all five sites are used.
        annotator: The choice of annotator group. Either 'expert' or 'non_expert'.
        rater: The id of a single rater of the annotator group. By default, the STAPLE consensus masks are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data, which contains the CT scan of a site once per structure.
        List of filepaths for the label data.
    """
    import nibabel as nib

    if site is not None and site not in SITES:
        raise ValueError(f"'{site}' is not a valid site. Choose one of {SITES}.")
    if annotator not in ANNOTATORS:
        raise ValueError(f"'{annotator}' is not a valid annotator group. Choose one of {list(ANNOTATORS)}.")

    data_dir = get_c3ro_data(path, download)

    raw_paths, label_paths = [], []
    for site_name in ([site] if site is not None else SITES):
        ct_path = os.path.join(data_dir, site_name, "CT", "NIFTI", f"Image_CT_{site_name}.nii.gz")
        seg_dir = os.path.join(data_dir, site_name, "Segmentations", ANNOTATORS[annotator])
        if rater is None:
            mask_paths = natsorted(glob(os.path.join(seg_dir, "Consensus", "*.nii.gz")))
        else:
            mask_paths = natsorted(glob(os.path.join(seg_dir, str(rater), "NIFTI", "*.nii.gz")))

        ct_shape = nib.load(ct_path).shape
        for mask_path in mask_paths:
            if nib.load(mask_path).shape == ct_shape:
                raw_paths.append(ct_path)
                label_paths.append(mask_path)

    if len(raw_paths) == 0:
        raise ValueError(f"No masks were found for site '{site}', annotator '{annotator}' and rater '{rater}'.")

    return raw_paths, label_paths


def get_c3ro_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    site: Optional[Literal["Breast", "GI", "GYN", "H&N", "Sarcoma"]] = None,
    annotator: Literal["expert", "non_expert"] = "expert",
    rater: Optional[str] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the C3RO dataset for organ and target volume segmentation in radiotherapy planning CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        site: The choice of cancer site. By default, all five sites are used.
        annotator: The choice of annotator group. Either 'expert' or 'non_expert'.
        rater: The id of a single rater of the annotator group. By default, the STAPLE consensus masks are used.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_c3ro_paths(path, site, annotator, rater, download)

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


def get_c3ro_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    site: Optional[Literal["Breast", "GI", "GYN", "H&N", "Sarcoma"]] = None,
    annotator: Literal["expert", "non_expert"] = "expert",
    rater: Optional[str] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the C3RO dataloader for organ and target volume segmentation in radiotherapy planning CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        site: The choice of cancer site. By default, all five sites are used.
        annotator: The choice of annotator group. Either 'expert' or 'non_expert'.
        rater: The id of a single rater of the annotator group. By default, the STAPLE consensus masks are used.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_c3ro_dataset(path, patch_shape, site, annotator, rater, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
