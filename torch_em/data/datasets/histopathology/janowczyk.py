"""The Janowczyk dataset contains annotations for nucleus, epithelium and tubule segmentation
in H&E stained histopathology images for breast cancer.

NOTE:
- The nuclei are sparsely annotated instances for ER+ breast cancer images.
- The epithelium and tubule are dense semantic annotations for breast cancer images.

The dataset is located at https://andrewjanowczyk.com/deep-learning/.
This dataset is from the publication https://doi.org/10.4103/2153-3539.186902.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import imageio.v3 as imageio
from skimage.measure import label as connected_components

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = {
    "nuclei": "https://andrewjanowczyk.com/wp-static/nuclei.tgz",
    "epithelium": "https://andrewjanowczyk.com/wp-static/epi.tgz",
    "tubule": "https://andrewjanowczyk.com/wp-static/tubule.tgz",
}

CHECKSUM = {
    "nuclei": "cb881c29d9f0ae5ad1d953160a4e00be70af329e0351eed614d51b4b66c65e6b",
    "epithelium": "5ac91a48de7d4f158f72cfc239b9a465849166397580b95d8f695095f54bcf6d",
    "tubule": "4f3e49d32b993c773a4d437f7483677d6b7c53a1d29f6b0b359a21722fa1f8f3",
}


def get_janowczyk_data(
    path: Union[os.PathLike, str],
    annotation: Literal['nuclei', 'epithelium', 'tubule'] = "nuclei",
    download: bool = False
) -> str:
    """Download the Janowczyk dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        annotation: The choice of annotated labels.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the dataset is downloaded.
    """
    if annotation not in ['nuclei', 'epithelium', 'tubule']:
        raise ValueError(f"'{annotation}' is not a supported annotation for labels.")

    data_dir = os.path.join(path, "data", annotation)
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    tar_path = os.path.join(path, f"{annotation}.tgz")
    util.download_source(
        path=tar_path, url=URL[annotation], download=download, checksum=CHECKSUM[annotation], verify=False
    )
    util.unzip_tarfile(tar_path=tar_path, dst=data_dir, remove=False)

    return data_dir


def get_janowczyk_paths(
    path: Union[os.PathLike, str],
    annotation: Literal['nuclei', 'epithelium', 'tubule'] = "nuclei",
    download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the Janowczyk data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        annotation: The choice of annotated labels.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_janowczyk_data(path, annotation, download)

    if annotation == "epithelium":
        label_paths = natsorted(glob(os.path.join(data_dir, "masks", "*_mask.png")))
        raw_paths = [p.replace("masks/", "").replace("_mask.png", ".tif") for p in label_paths]
    elif annotation == "tubule":
        label_paths = natsorted(glob(os.path.join(data_dir, "*_anno.bmp")))
        raw_paths = [p.replace("_anno", "") for p in label_paths]
    else:  # nuclei
        raw_paths = natsorted(glob(os.path.join(data_dir, "*_original.tif")))
        label_paths = []
        for lpath in tqdm(glob(os.path.join(data_dir, "*_mask.png")), desc="Preprocessing 'nuclei' labels"):
            neu_label_path = lpath.replace("_mask.png", "_preprocessed_labels.tif")
            label_paths.append(neu_label_path)
            if os.path.exists(neu_label_path):
                continue

            label = imageio.imread(lpath)
            label = connected_components(label)
            imageio.imwrite(neu_label_path, label, compression="zlib")
        label_paths = natsorted(label_paths)

    return raw_paths, label_paths


def get_janowczyk_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    annotation: Literal['nuclei', 'epithelium', 'tubule'] = "nuclei",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Janowczyk dataset for nucleus, epithelium and tubule segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        annotation: The choice of annotated labels.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_janowczyk_paths(path, annotation, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        is_seg_dataset=False,
        with_channels=True,
        ndim=2,
        patch_shape=patch_shape,
        **kwargs
    )


def get_janowczyk_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    annotation: Literal['nuclei', 'epithelium', 'tubule'] = "nuclei",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Janowczyk dataloader for nucleus, epithelium and tubule segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        annotation: The choice of annotated labels.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_janowczyk_dataset(path, patch_shape, annotation, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
