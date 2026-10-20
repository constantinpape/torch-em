"""This dataset contains nucleus instance segmentations in volume electron microscopy (vEM) data.

It has 9 2D array tomography (AT) SEM images and 2 3D FIB-SEM volumes of cancer cells and tumor tissue
(human breast cancer cells, mouse lung tumor, mouse 4T1 cells and human MB-231 cells treated with ONC-201),
with paired nucleus instance labels. The 2D labels were predicted with the NucleoNet model and manually proofread,
the 3D labels (a single nucleus per volume) were obtained by 2D inference in orthogonal planes, voting and proofreading.

Nuclei touching the image border are removed from the labels of most 2D images and kept in others,
see `EDGE_NUCLEI_INCLUDED`. The 2D images are very large (up to 49530 x 7448 pixels).

The record lists a 10th 2D image ('id_9'), but its file in the archive is truncated (4096 bytes instead of 100 MB),
so it is not available here.

The data is available at https://www.ebi.ac.uk/biostudies/studies/S-BIAD2822 (CC-BY-4.0).
The record does not list a publication. Please cite the BioImage Archive entry S-BIAD2822 if you use this dataset.
"""

import os
from typing import List, Literal, Optional, Sequence, Tuple, Union

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://www.ebi.ac.uk/biostudies/files/S-BIAD2822"

SAMPLES = {
    1: ("2d", "img_2bf732c1.tif", "lbl_2bf732c1.tif"),
    2: ("2d", "img_97dde2ee.tiff", "lbl_97dde2ee.tif"),
    3: ("2d", "img_626dde70.tif", "lbl_626dde70.tif"),
    4: ("2d", "img_57f48663.tif", "lbl_57f48663.tif"),
    5: ("2d", "img_22668234.tiff", "lbl_22668234.tif"),
    6: ("2d", "img_c69fbe8a.tif", "lbl_c69fbe8a.tif"),
    7: ("2d", "img_4ff7ad90.tif", "lbl_4ff7ad90.tif"),
    8: ("2d", "img_ee855e08.tif", "lbl_ee855e08.tif"),
    10: ("2d", "img_87356f94.tif", "lbl_87356f94.tif"),
    11: ("3d", "img_4808a925.tif", "lbl_4808a925.tif"),
    12: ("3d", "img_596d5ecb.tif", "lbl_596d5ecb.tif"),
}

# The 2D images in which the nuclei touching the image border are kept in the labels.
EDGE_NUCLEI_INCLUDED = (8, 10)

CHECKSUMS = {
    "id_1/img_2bf732c1.tif": "d7705c8a7768738a4a685f15fe998c0d1832090bcd316a7c4564cc8392b05328",
    "id_1/lbl_2bf732c1.tif": "55660f4a1093877da8c330937498efcfea542e0c7b4cf5690ae80ac84182bcaf",
    "id_2/img_97dde2ee.tiff": "9753295d17fe82e6d91d87a2f49625b48fabceea35798263a8cb918c840f8120",
    "id_2/lbl_97dde2ee.tif": "f093ab210d0e64aa27040178de3a3a47373dc3d4df342efab8143920bafb2dc7",
    "id_3/img_626dde70.tif": "fa0ce1ef7cf3623a5f0bf8e74d0b40d482359244abb1168c329350d81af5c10e",
    "id_3/lbl_626dde70.tif": "8046479b75eb3a854eff34a4b8ef9906219a8aa7f15b18be7d3834111244bcda",
    "id_4/img_57f48663.tif": "0e02b8349e05b407afb25c643d16b21d7ead570b2389e6ebd0628abb3b2ccaa8",
    "id_4/lbl_57f48663.tif": "a63c7efbb12aa514c81aa6f2c00cfb5d1d7092eacc868ba9b496d8b46b8e0e30",
    "id_5/img_22668234.tiff": "fa7d6f4b75017317c42418e021581060dbc8bf7e6c4b14374adf0bf673ebb6a7",
    "id_5/lbl_22668234.tif": "861ded58e610ca45d9162d5a9f99147611a8deb728bdfa611cde8477b9dc23af",
    "id_6/img_c69fbe8a.tif": "abf005036147a765a2212ac5c00055b32aa1591db41db871bd5880d00926dacc",
    "id_6/lbl_c69fbe8a.tif": "edc06af9061744c0993297a616258a000c32e6727fa16dbef1b1195ae883e88d",
    "id_7/img_4ff7ad90.tif": "e5032ffe2bd3f6f12638a3f25409b682cda410df780dc44224d203c68106d6c7",
    "id_7/lbl_4ff7ad90.tif": "b9cc9e2868c0775df3fef0b587aa05ca4d6904a85015dd5ed6f94f044fa30248",
    "id_8/img_ee855e08.tif": "2f85a64b60f47530d24c43baa0b7f727516fb87ee7d6ebcf84d338c685bc9c06",
    "id_8/lbl_ee855e08.tif": "2dc89dc2a051c74cd366c96861f6d1dda07d0774eec5db4dfcae5075a147bf59",
    "id_10/img_87356f94.tif": "5e39af57acc6b69b979f587c896b7e7617d7d790cd5d598c22ca553565104127",
    "id_10/lbl_87356f94.tif": "788179894654f4488cf404856752269b4e7402d15353e83d01c259de251f2cc6",
    "id_11/img_4808a925.tif": "2978908ebf14933e906c826dcf836831b6f9707dee49abc250876556e209613f",
    "id_11/lbl_4808a925.tif": "c253fea05e70cb6229d65ea80f04101e3be8b7f1f07ea235d1d914b4d7dfb87a",
    "id_12/img_596d5ecb.tif": "17505809d0317e3278ce40cd11a787af089f3727fa23bc4b6b793693653031ea",
    "id_12/lbl_596d5ecb.tif": "6245dc72027876f1f00380840ea176313a6a87386ccb27d69e18288a63e187f4",
}


def _select_sample_ids(dimensionality, sample_ids):
    if dimensionality not in ("2d", "3d"):
        raise ValueError(f"'{dimensionality}' is not a valid dimensionality. Choose from '2d' or '3d'.")

    available = [sample_id for sample_id, (dim, _, _) in SAMPLES.items() if dim == dimensionality]
    if sample_ids is None:
        return available

    sample_ids = [sample_ids] if isinstance(sample_ids, int) else list(sample_ids)
    invalid = [sample_id for sample_id in sample_ids if sample_id not in available]
    if invalid:
        raise ValueError(f"The sample ids {invalid} are not available for '{dimensionality}'. Choose from {available}.")
    return sorted(sample_ids)


def get_vem_nuclei_data(
    path: Union[os.PathLike, str], sample_ids: Optional[Sequence[int]] = None, download: bool = False
) -> str:
    """Download the vEM nuclei data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        sample_ids: The ids of the samples to download, see `SAMPLES`. By default all samples are downloaded
            (about 1.8 GB).
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder with the data.
    """
    sample_ids = list(SAMPLES) if sample_ids is None else list(sample_ids)
    for sample_id in sample_ids:
        _, img_name, lbl_name = SAMPLES[sample_id]
        for name in (img_name, lbl_name):
            rel_path = f"id_{sample_id}/{name}"
            file_path = os.path.join(path, rel_path)
            if not os.path.exists(file_path):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
            util.download_source(file_path, f"{BASE_URL}/{rel_path}", download, checksum=CHECKSUMS[rel_path])
    return str(path)


def get_vem_nuclei_paths(
    path: Union[os.PathLike, str],
    dimensionality: Literal["2d", "3d"] = "2d",
    sample_ids: Optional[Union[int, Sequence[int]]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the vEM nuclei data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        dimensionality: The data to use, either '2d' (array tomography images with ids 1-8 and 10)
            or '3d' (FIB-SEM volumes with ids 11 and 12).
        sample_ids: The samples to use. By default all samples of the dimensionality are used.
        download: Whether to download the data if it is not present.

    Returns:
        The filepaths to the images.
        The filepaths to the labels.
    """
    sample_ids = _select_sample_ids(dimensionality, sample_ids)
    get_vem_nuclei_data(path, sample_ids, download)

    raw_paths = [os.path.join(path, f"id_{sample_id}", SAMPLES[sample_id][1]) for sample_id in sample_ids]
    label_paths = [os.path.join(path, f"id_{sample_id}", SAMPLES[sample_id][2]) for sample_id in sample_ids]
    return raw_paths, label_paths


def get_vem_nuclei_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    dimensionality: Literal["2d", "3d"] = "2d",
    sample_ids: Optional[Union[int, Sequence[int]]] = None,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    **kwargs
) -> Dataset:
    """Get the vEM nuclei dataset for nucleus segmentation in EM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training. Use (H, W) for '2d' and (D, H, W) for '3d'.
        dimensionality: The data to use, either '2d' (array tomography images) or '3d' (FIB-SEM volumes).
        sample_ids: The samples to use. By default all samples of the dimensionality are used.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to return a binary segmentation target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == (2 if dimensionality == "2d" else 3), f"{dimensionality}, {patch_shape}"

    raw_paths, label_paths = get_vem_nuclei_paths(path, dimensionality, sample_ids, download)

    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, binary=binary, boundaries=boundaries, offsets=offsets
    )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths, raw_key=None, label_paths=label_paths, label_key=None, patch_shape=patch_shape, **kwargs
    )


def get_vem_nuclei_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    dimensionality: Literal["2d", "3d"] = "2d",
    sample_ids: Optional[Union[int, Sequence[int]]] = None,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    **kwargs
) -> DataLoader:
    """Get the vEM nuclei dataloader for nucleus segmentation in EM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training. Use (H, W) for '2d' and (D, H, W) for '3d'.
        dimensionality: The data to use, either '2d' (array tomography images) or '3d' (FIB-SEM volumes).
        sample_ids: The samples to use. By default all samples of the dimensionality are used.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to return a binary segmentation target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_vem_nuclei_dataset(
        path=path, patch_shape=patch_shape, dimensionality=dimensionality, sample_ids=sample_ids, download=download,
        offsets=offsets, boundaries=boundaries, binary=binary, **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
