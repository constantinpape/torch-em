"""The Cell Tracking Challenge contains annotated data for cell segmentation and tracking.
We currently provide the 2d datasets with segmentation annotations.

If you use this data in your research please cite https://doi.org/10.1038/nmeth.4473.
"""

import os
from glob import glob
from shutil import copyfile
from typing import Optional, Tuple, Union

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


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
    }
}


def _get_ctc_url_and_checksum(dataset_name, split):
    if split == "train":
        _link_to_split = "training-datasets"
    else:
        _link_to_split = "test-datasets"

    url = f"http://data.celltrackingchallenge.net/{_link_to_split}/{dataset_name}.zip"
    checksum = CTC_CHECKSUMS[split][dataset_name]
    return url, checksum


def get_ctc_segmentation_data(
    path: Union[os.PathLike, str], dataset_name: str, split: str, download: bool = False,
) -> str:
    f"""Download training data from the Cell Tracking Challenge.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        dataset_name: Name of the dataset to be downloaded. The available datasets are:
            {', '.join(CTC_CHECKSUMS['train'].keys())}
        split: The split to download. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the training data.
    """
    dataset_names = list(CTC_CHECKSUMS["train"].keys())
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


def _require_gt_images(data_path, vol_ids):
    image_paths, label_paths = [], []

    if isinstance(vol_ids, str):
        vol_ids = [vol_ids]

    for vol_id in vol_ids:
        image_folder = os.path.join(data_path, vol_id)
        assert os.path.join(image_folder), f"Cannot find volume id, {vol_id} in {data_path}."

        label_folder = os.path.join(data_path, f"{vol_id}_GT", "SEG")

        # copy over the images corresponding to the labeled frames
        label_image_folder = os.path.join(data_path, f"{vol_id}_GT", "IM")
        os.makedirs(label_image_folder, exist_ok=True)

        this_label_paths = glob(os.path.join(label_folder, "*.tif"))
        for label_path in this_label_paths:
            fname = os.path.basename(label_path)
            image_label_path = os.path.join(label_image_folder, fname)
            if not os.path.exists(image_label_path):
                im_name = "t" + fname.lstrip("main_seg")
                image_path = os.path.join(image_folder, im_name)
                assert os.path.join(image_path), image_path
                copyfile(image_path, image_label_path)

        image_paths.append(label_image_folder)
        label_paths.append(label_folder)

    return image_paths, label_paths


def get_ctc_segmentation_paths(
    path: Union[os.PathLike, str],
    dataset_name: str,
    split: str = "train",
    vol_id: Optional[int] = None,
    download: bool = False,
) -> Tuple[str, str]:
    f"""Get paths to the Cell Tracking Challenge data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        dataset_name: Name of the dataset to be downloaded. The available datasets are:
            {', '.join(CTC_CHECKSUMS['train'].keys())}
        split: The split to download. Currently only supports 'train'.
        vol_id: The train id to load.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder where image data is stored.
        Filepath to the folder where label data is stored.
    """
    data_path = get_ctc_segmentation_data(path, dataset_name, split, download)

    if vol_id is None:
        vol_ids = glob(os.path.join(data_path, "*_GT"))
        vol_ids = [os.path.basename(vol_id) for vol_id in vol_ids]
        vol_ids = [vol_id.rstrip("_GT") for vol_id in vol_ids]
    else:
        vol_ids = vol_id

    image_path, label_path = _require_gt_images(data_path, vol_ids)
    return image_path, label_path


def get_ctc_segmentation_dataset(
    path: Union[os.PathLike, str],
    dataset_name: str,
    patch_shape: Tuple[int, int, int],
    split: str = "train",
    vol_id: Optional[int] = None,
    download: bool = False,
    **kwargs,
) -> Dataset:
    f"""Get the CTC dataset for cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        dataset_name: Name of the dataset to be downloaded. The available datasets are:
            {', '.join(CTC_CHECKSUMS['train'].keys())}
        patch_shape: The patch shape to use for training.
        split: The split to download. Currently only supports 'train'.
        vol_id: The train id to load.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
       The segmentation dataset.
    """
    assert split in ["train"]

    image_path, label_path = get_ctc_segmentation_paths(path, dataset_name, split, vol_id, download)

    kwargs = util.update_kwargs(kwargs, "ndim", 2)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_path,
        raw_key="*.tif",
        label_paths=label_path,
        label_key="*.tif",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_ctc_segmentation_loader(
    path: Union[os.PathLike, str],
    dataset_name: str,
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    split: str = "train",
    vol_id: Optional[int] = None,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    f"""Get the CTC dataloader for cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        dataset_name: Name of the dataset to be downloaded. The available datasets are:
            {', '.join(CTC_CHECKSUMS['train'].keys())}
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        split: The split to download. Currently only supports 'train'.
        vol_id: The train id to load.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
       The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_ctc_segmentation_dataset(path, dataset_name, patch_shape, split, vol_id, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
