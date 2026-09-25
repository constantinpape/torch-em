"""The Cell Tracking Challenge contains annotated data for cell segmentation and tracking.
We provide the 2d and 3d datasets with segmentation annotations. The 2d datasets are listed in `CTC_2D_DATASETS`,
the 3d datasets in `CTC_3D_DATASETS`. See https://celltrackingchallenge.net/2d-datasets/ and
https://celltrackingchallenge.net/3d-datasets/ for details on the individual datasets.

The segmentation annotations are sparse: only some time points are annotated, and for some of the 3d datasets only
individual slices of a time point are annotated. Time points with a fully annotated volume are loaded as 3d data.
If a 3d dataset only has slice-wise annotations, the annotated slices are extracted from the raw volumes and loaded
as 2d data instead. In addition to the manually curated gold truth annotations ('GT'), the challenge provides
silver truth annotations ('ST') that are computationally derived and cover all time points of the 2d and 3d datasets.

If you use this data in your research please cite https://doi.org/10.1038/nmeth.4473.
"""

import os
import re
from glob import glob
from shutil import copyfile
from typing import List, Literal, Optional, Tuple, Union

import imageio.v3 as imageio
from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


CTC_2D_DATASETS = [
    "BF-C2DL-HSC", "BF-C2DL-MuSC", "DIC-C2DH-HeLa", "Fluo-C2DL-Huh7", "Fluo-C2DL-MSC",
    "Fluo-N2DH-GOWT1", "Fluo-N2DH-SIM+", "Fluo-N2DL-HeLa", "PhC-C2DH-U373", "PhC-C2DL-PSC",
]
"""The names of the 2d datasets."""

CTC_3D_DATASETS = [
    "Fluo-C3DH-A549", "Fluo-C3DH-A549-SIM", "Fluo-C3DH-H157", "Fluo-C3DL-MDA231", "Fluo-N3DH-CE",
    "Fluo-N3DH-CHO", "Fluo-N3DH-SIM+", "Fluo-N3DL-DRO", "Fluo-N3DL-TRIC", "Fluo-N3DL-TRIF",
]
"""The names of the 3d datasets."""

CTC_CHECKSUMS = {
    "train": {
        "BF-C2DL-HSC": "0aa68ec37a9b06e72a5dfa07d809f56e1775157fb674bb75ff904936149657b1",
        "BF-C2DL-MuSC": "ca72b59042809120578a198ba236e5ed3504dd6a122ef969428b7c64f0a5e67d",
        "DIC-C2DH-HeLa": "832fed2d05bb7488cf9c51a2994b75f8f3f53b3c3098856211f2d39023c34e1a",
        "Fluo-C2DL-Huh7": "1912658c1b3d8b38b314eb658b559e7b39c256917150e9b3dd8bfdc77347617d",
        "Fluo-C2DL-MSC": "a083521f0cb673ae02d4957c5e6580c2e021943ef88101f6a2f61b944d671af2",
        "Fluo-N2DH-GOWT1": "1a7bd9a7d1d10c4122c7782427b437246fb69cc3322a975485c04e206f64fc2c",
        "Fluo-N2DH-SIM+": "3e809148c87ace80c72f563b56c35e0d9448dcdeb461a09c83f61e93f5e40ec8",
        "Fluo-N2DL-HeLa": "35dd99d58e071aba0b03880128d920bd1c063783cc280f9531fbdc5be614c82e",
        "PhC-C2DH-U373": "b18185c18fce54e8eeb93e4bbb9b201d757add9409bbf2283b8114185a11bc9e",
        "PhC-C2DL-PSC": "9d54bb8febc8798934a21bf92e05d92f5e8557c87e28834b2832591cdda78422",
        "Fluo-C3DH-A549": "46be7a5403f98070218414e5b71a302f29f697ffcae68f45ecaea5d737076026",
        "Fluo-C3DH-A549-SIM": "321bd505854a4e3c9b8255cb903ecc48f43f14abd444448ef6945fc1ec9fa4cf",
        "Fluo-C3DH-H157": "9540984397dfb7129b5bff8177b7da11c497708421e6de2713c186b5b15a3c9b",
        "Fluo-C3DL-MDA231": "b1044eeaac644f1abfbf91d9b6c97bebfb02f4dee2b2a4aed60d0039a375aa84",
        "Fluo-N3DH-CE": "eb3d37cacb3b51d3a427a0c43deaaf1d60cf2eba009f12569e720c2c2fd1e04c",
        "Fluo-N3DH-CHO": "48d7e32b6408dddd04f1b6e4153e91f19181b405f4c75610629a25e05d40fe77",
        "Fluo-N3DH-SIM+": "f874a297a97ba2b144f2cbe9f06f66f57996251e89b17f7d5ef96a5d99d0fd11",
        "Fluo-N3DL-DRO": "57c3039b746116fa068e97f25bb14e8deaf3bd11528d61709c5ae31bc7fc7f11",
        "Fluo-N3DL-TRIC": "225f0ed3ba706d9c28101b01bd22ad96974ce2d2334d5c2bf7cddbf2236ec6bb",
        # The Fluo-N3DL-TRIF training data is 320 GB large, its checksum has not been computed yet.
    },
    "test": {
        "BF-C2DL-HSC": "fd1c05ec625fd0526c8369d1139babe137e885457eee98c10d957da578d0d5bc",
        "BF-C2DL-MuSC": "c5cae259e6090e82a2596967fb54c8a768717c1772398f8546ad1c8df0820450",
        "DIC-C2DH-HeLa": "5e5d5f2aa90aef99d750cf03f5c12d799d50b892f98c86950e07a2c5955ac01f",
        "Fluo-C2DL-Huh7": "cc7359f8fb6b0c43995365e83ce0116d32f477ac644b2ca02b98bc253e2bcbbe",
        "Fluo-C2DL-MSC": "c90b13e603dde52f17801d4f0cadde04ed7f21cc05296b1f0957d92dbfc8ffa6",
        "Fluo-N2DH-GOWT1": "c6893ec2d63459de49d4dc21009b04275573403c62cc02e6ee8d0cb1a5068add",
        "Fluo-N2DH-SIM+": "c4f257add739b284d02176057814de345dee2ac1a7438e360ccd2df73618db68",
        "Fluo-N2DL-HeLa": "45cf3daf05e8495aa2ce0febacca4cf0928fab808c0b14ed2eb7289a819e6bb8",
        "PhC-C2DH-U373": "7aa3162e4363a416b259149adc13c9b09cb8aecfe8165eb1428dd534b66bec8a",
        "PhC-C2DL-PSC": "8c98ac6203e7490157ceb6aa1131d60a3863001b61fb75e784bc49d47ee264d5",
        # The checksums for the test data of the 3d datasets have not been computed yet.
    }
}
"""The checksums of the zip archives for the train and test split of the datasets."""


def _get_ctc_url_and_checksum(dataset_name, split):
    if split == "train":
        _link_to_split = "training-datasets"
    else:
        _link_to_split = "test-datasets"

    url = f"http://data.celltrackingchallenge.net/{_link_to_split}/{dataset_name}.zip"
    # The checksum verification is skipped for the archives whose checksum has not been computed yet.
    checksum = CTC_CHECKSUMS[split].get(dataset_name)
    return url, checksum


def get_ctc_segmentation_data(
    path: Union[os.PathLike, str], dataset_name: str, split: str, download: bool = False,
) -> str:
    """Download training data from the Cell Tracking Challenge.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        dataset_name: Name of the dataset to be downloaded. The available datasets are listed in
            `CTC_2D_DATASETS` and `CTC_3D_DATASETS`.
        split: The split to download. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the training data.
    """
    dataset_names = CTC_2D_DATASETS + CTC_3D_DATASETS
    if dataset_name not in dataset_names:
        raise ValueError(f"Invalid dataset: {dataset_name}, choose one of {dataset_names}.")

    data_path = os.path.join(path, split, dataset_name)

    if os.path.exists(data_path):
        return data_path

    os.makedirs(data_path)
    url, checksum = _get_ctc_url_and_checksum(dataset_name, split)
    zip_path = os.path.join(path, f"{dataset_name}.zip")
    util.download_source(zip_path, url, download, checksum=checksum)
    util.unzip(zip_path, os.path.join(path, split), remove=True)

    return data_path


def _parse_label_name(fname):
    # Annotations of a full time point are named 'man_seg<T>.tif',
    # annotations of a single slice of a time point are named 'man_seg_<T>_<Z>.tif'.
    match = re.fullmatch(r"man_seg(\d+)\.tif", fname)
    if match is not None:
        return match.group(1), None
    match = re.fullmatch(r"man_seg_(\d+)_(\d+)\.tif", fname)
    if match is not None:
        return match.group(1), int(match.group(2))
    raise ValueError(f"Unexpected name for a segmentation annotation: {fname}")


def _require_gt_image(image_folder, label_path, label_image_folder):
    fname = os.path.basename(label_path)
    image_label_path = os.path.join(label_image_folder, fname)
    if os.path.exists(image_label_path):
        return image_label_path

    time_point, slice_id = _parse_label_name(fname)
    image_path = os.path.join(image_folder, f"t{time_point}.tif")
    assert os.path.exists(image_path), image_path

    if slice_id is None:
        # Copy over the image corresponding to the fully labeled time point.
        copyfile(image_path, image_label_path)
    else:
        # Extract the labeled slice from the image volume.
        image = imageio.imread(image_path)
        imageio.imwrite(image_label_path, image[slice_id])

    return image_label_path


def _require_gt_images(data_path, vol_ids, annotation_type="GT", return_files=False):
    image_paths, label_paths = [], []

    if isinstance(vol_ids, str):
        vol_ids = [vol_ids]

    # Check whether any of the time points is fully annotated. If so, the fully annotated time points are used.
    # Otherwise the slice-wise annotations are used.
    all_label_paths = {
        vol_id: sorted(glob(os.path.join(data_path, f"{vol_id}_{annotation_type}", "SEG", "*.tif")))
        for vol_id in vol_ids
    }
    assert any(len(paths) > 0 for paths in all_label_paths.values()), f"No annotations found in {data_path}."
    use_slices = all(
        _parse_label_name(os.path.basename(p))[1] is not None for paths in all_label_paths.values() for p in paths
    )

    for vol_id in vol_ids:
        image_folder = os.path.join(data_path, vol_id)
        assert os.path.exists(image_folder), f"Cannot find volume id, {vol_id} in {data_path}."

        label_folder = os.path.join(data_path, f"{vol_id}_{annotation_type}", "SEG")

        # Copy over the images corresponding to the labeled frames.
        label_image_folder = os.path.join(data_path, f"{vol_id}_{annotation_type}", "IM")
        os.makedirs(label_image_folder, exist_ok=True)

        this_label_paths = [
            p for p in all_label_paths[vol_id] if (_parse_label_name(os.path.basename(p))[1] is not None) == use_slices
        ]
        this_image_paths = [_require_gt_image(image_folder, p, label_image_folder) for p in this_label_paths]

        if return_files:
            image_paths.extend(this_image_paths)
            label_paths.extend(this_label_paths)
        else:
            image_paths.append(label_image_folder)
            label_paths.append(label_folder)

    return image_paths, label_paths


def get_ctc_segmentation_paths(
    path: Union[os.PathLike, str],
    dataset_name: str,
    split: str = "train",
    vol_id: Optional[int] = None,
    download: bool = False,
    annotation_type: Literal["GT", "ST"] = "GT",
) -> Tuple[List[str], List[str]]:
    """Get paths to the Cell Tracking Challenge data.

    For the 2d datasets this returns the folders with the images of the annotated time points and the folders
    with the corresponding annotations. For the 3d datasets this returns the filepaths of the annotated volumes
    (or the annotated slices for datasets that only have slice-wise annotations) and of the corresponding annotations.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        dataset_name: Name of the dataset to be downloaded. The available datasets are listed in
            `CTC_2D_DATASETS` and `CTC_3D_DATASETS`.
        split: The split to download. Currently only supports 'train'.
        vol_id: The train id to load.
        download: Whether to download the data if it is not present.
        annotation_type: The type of annotations to load. Either 'GT' for the manually curated gold truth
            annotations or 'ST' for the computationally derived silver truth annotations.

    Returns:
        Filepaths to the image data.
        Filepaths to the label data.
    """
    data_path = get_ctc_segmentation_data(path, dataset_name, split, download)

    if vol_id is None:
        vol_ids = glob(os.path.join(data_path, "*_GT"))
        vol_ids = [os.path.basename(vol_id) for vol_id in vol_ids]
        vol_ids = sorted(vol_id[:-len("_GT")] for vol_id in vol_ids)
    else:
        vol_ids = vol_id

    assert annotation_type in ("GT", "ST"), f"Invalid annotation type: {annotation_type}, choose 'GT' or 'ST'."
    return_files = dataset_name in CTC_3D_DATASETS
    image_path, label_path = _require_gt_images(data_path, vol_ids, annotation_type, return_files=return_files)
    return image_path, label_path


def get_ctc_segmentation_dataset(
    path: Union[os.PathLike, str],
    dataset_name: str,
    patch_shape: Tuple[int, ...],
    split: str = "train",
    vol_id: Optional[int] = None,
    download: bool = False,
    annotation_type: Literal["GT", "ST"] = "GT",
    **kwargs,
) -> Dataset:
    """Get the CTC dataset for cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        dataset_name: Name of the dataset to be downloaded. The available datasets are listed in
            `CTC_2D_DATASETS` and `CTC_3D_DATASETS`.
        patch_shape: The patch shape to use for training. Use a 2d patch shape (with a leading 1) for the 2d datasets
            and a 3d patch shape for the 3d datasets (a 2d patch shape for the 3d datasets with slice-wise annotations).
        split: The split to download. Currently only supports 'train'.
        vol_id: The train id to load.
        download: Whether to download the data if it is not present.
        annotation_type: The type of annotations to load. Either 'GT' for the manually curated gold truth
            annotations or 'ST' for the computationally derived silver truth annotations.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
       The segmentation dataset.
    """
    assert split in ["train"]

    image_path, label_path = get_ctc_segmentation_paths(path, dataset_name, split, vol_id, download, annotation_type)

    if dataset_name in CTC_3D_DATASETS:
        # The data is loaded as 3d if it has fully annotated time points and as 2d if it only has annotated slices.
        ndim = 2 if _parse_label_name(os.path.basename(label_path[0]))[1] is not None else 3
        kwargs = util.update_kwargs(kwargs, "ndim", ndim)
        raw_key, label_key = None, None
    else:
        kwargs = util.update_kwargs(kwargs, "ndim", 2)
        raw_key, label_key = "*.tif", "*.tif"

    return torch_em.default_segmentation_dataset(
        raw_paths=image_path,
        raw_key=raw_key,
        label_paths=label_path,
        label_key=label_key,
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_ctc_segmentation_loader(
    path: Union[os.PathLike, str],
    dataset_name: str,
    patch_shape: Tuple[int, ...],
    batch_size: int,
    split: str = "train",
    vol_id: Optional[int] = None,
    download: bool = False,
    annotation_type: Literal["GT", "ST"] = "GT",
    **kwargs,
) -> DataLoader:
    """Get the CTC dataloader for cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        dataset_name: Name of the dataset to be downloaded. The available datasets are listed in
            `CTC_2D_DATASETS` and `CTC_3D_DATASETS`.
        patch_shape: The patch shape to use for training. Use a 2d patch shape (with a leading 1) for the 2d datasets
            and a 3d patch shape for the 3d datasets (a 2d patch shape for the 3d datasets with slice-wise annotations).
        batch_size: The batch size for training.
        split: The split to download. Currently only supports 'train'.
        vol_id: The train id to load.
        download: Whether to download the data if it is not present.
        annotation_type: The type of annotations to load. Either 'GT' for the manually curated gold truth
            annotations or 'ST' for the computationally derived silver truth annotations.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
       The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_ctc_segmentation_dataset(
        path, dataset_name, patch_shape, split, vol_id, download, annotation_type, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
