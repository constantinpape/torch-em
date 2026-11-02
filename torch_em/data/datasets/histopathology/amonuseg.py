"""The AMONuSeg dataset contains nucleus annotations for H&E stained images from Nigeria.

The dataset holds 48 images of 1280 x 960 pixels, taken with a microscope camera, with 12 images each of breast,
cervical, inguinal (lymph node) and skin tissue. Two experts annotated the nuclei: 'exp1' and 'exp2'.
Both annotators corrected draft masks from Fiji by hand, and four pathologists reviewed and validated them.
NOTE: The masks are semantic (nucleus vs. background), not instance masks. This module creates the instances as
connected components, so touching nuclei merge into one instance.
NOTE: Some mask files are stored with lossy compression artifacts, so this module binarizes all masks at half
the intensity range before it labels them.

The dataset is located at https://github.com/zerouaoui/AMONUSEG (MIT license).
This dataset is from the publication https://doi.org/10.1007/978-3-031-72114-4_10.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Literal, Optional, Sequence

import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


# The commit of the repository that this module downloads, so that the data does not change.
COMMIT = "5614bf5c0aff11b3bf66420af39502d734e1c857"
RAW_URL = f"https://raw.githubusercontent.com/zerouaoui/AMONUSEG/{COMMIT}"
TREE_URL = f"https://api.github.com/repos/zerouaoui/AMONUSEG/git/trees/{COMMIT}?recursive=1"
ORGANS = ("breast", "cervical", "inguinal", "skin")
# The folders of the images and of the masks of both experts, per organ. The repository spells them differently.
FOLDERS = {
    "breast": ("Breast", "Breast Masks", "Brest Masks E2"),
    "cervical": ("Cervical", "Cervical Masks", "Cervical Masks E2"),
    "inguinal": ("Inguinal", "Inguinal Masks", "Inguinal Masks E2"),
    "skin": ("Skin", "Skin Masks", "Skin Masks E2"),
}


def _download_files(path, download):
    import json
    from urllib.parse import quote

    tree_path = os.path.join(path, "tree.json")
    util.download_source(tree_path, TREE_URL, download, checksum=None)
    with open(tree_path) as f:
        files = [x["path"] for x in json.load(f)["tree"] if x["type"] == "blob" and x["path"].startswith("AMONuSeg/")]
    for name in files:
        local = os.path.join(path, name)
        os.makedirs(os.path.dirname(local), exist_ok=True)
        util.download_source(local, f"{RAW_URL}/{quote(name)}", download, checksum=None)


def _find_mask(mask_dir, stem):
    # The mask of image '<stem>.jpg' is '<stem>.png' or '<stem>-Mask-EXP1.png'.
    candidates = glob(os.path.join(mask_dir, f"{stem}.*")) + glob(os.path.join(mask_dir, f"{stem}-*"))
    assert len(candidates) == 1, (mask_dir, stem, candidates)
    return candidates[0]


def _to_instances(mask_path):
    from skimage.measure import label

    mask = imageio.imread(mask_path)
    mask = mask[..., 0] if mask.ndim == 3 else mask
    return label(mask > (mask.max() / 2)).astype("uint32")


def _create_h5_files(path):
    import h5py

    for organ in ORGANS:
        image_dir, exp1_dir, exp2_dir = (os.path.join(path, "AMONuSeg", folder) for folder in FOLDERS[organ])
        out_dir = os.path.join(path, "preprocessed", organ)
        os.makedirs(out_dir, exist_ok=True)
        for image_path in tqdm(natsorted(glob(os.path.join(image_dir, "*.jpg"))), desc=f"Preprocess {organ} images"):
            stem = os.path.splitext(os.path.basename(image_path))[0]
            image = imageio.imread(image_path)[..., :3]
            exp1 = _to_instances(_find_mask(exp1_dir, stem))
            exp2 = _to_instances(_find_mask(exp2_dir, stem))
            assert image.shape[:2] == exp1.shape == exp2.shape, stem

            with h5py.File(os.path.join(out_dir, f"{stem}.h5"), "w") as f:
                f.create_dataset("raw", data=image.transpose(2, 0, 1), compression="gzip")
                f.create_dataset("labels/instances/exp1", data=exp1, compression="gzip")
                f.create_dataset("labels/instances/exp2", data=exp2, compression="gzip")


def get_amonuseg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the AMONuSeg dataset.

    Args:
        path: Filepath to a folder where the downloaded data is saved.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    data_dir = os.path.join(path, "preprocessed")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    _download_files(path, download)
    _create_h5_files(path)
    return data_dir


def get_amonuseg_paths(
    path: Union[os.PathLike, str], organs: Optional[Sequence[str]] = None, download: bool = False
) -> List[str]:
    """Get paths to the AMONuSeg data.

    Args:
        path: Filepath to a folder where the downloaded data is saved.
        organs: The organs, out of 'breast', 'cervical', 'inguinal' and 'skin'. By default, all organs are returned.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the input data.
    """
    data_dir = get_amonuseg_data(path, download)
    organs = ORGANS if organs is None else organs
    assert all(organ in ORGANS for organ in organs), organs
    return [p for organ in organs for p in natsorted(glob(os.path.join(data_dir, organ, "*.h5")))]


def get_amonuseg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    organs: Optional[Sequence[str]] = None,
    annotator: Literal["exp1", "exp2"] = "exp1",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the AMONuSeg dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data is saved.
        patch_shape: The patch shape to use for training.
        organs: The organs, out of 'breast', 'cervical', 'inguinal' and 'skin'. By default, all organs are used.
        annotator: The annotations of the first ('exp1') or the second ('exp2') expert.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert annotator in ("exp1", "exp2"), annotator
    paths = get_amonuseg_paths(path, organs, download)
    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key=f"labels/instances/{annotator}",
        patch_shape=patch_shape,
        ndim=2,
        with_channels=True,
        **kwargs
    )


def get_amonuseg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    organs: Optional[Sequence[str]] = None,
    annotator: Literal["exp1", "exp2"] = "exp1",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the AMONuSeg dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data is saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        organs: The organs, out of 'breast', 'cervical', 'inguinal' and 'skin'. By default, all organs are used.
        annotator: The annotations of the first ('exp1') or the second ('exp2') expert.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_amonuseg_dataset(path, patch_shape, organs, annotator, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
