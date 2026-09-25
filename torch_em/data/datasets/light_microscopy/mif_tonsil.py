"""The Multiplex IF Tonsil dataset contains annotations for nucleus and whole-cell instance segmentation
in multiplex immunofluorescence images of human tonsil tissue.

The dataset consists of 10 image regions (100x100 pixels, 7 channels: CD21, CD23, CD20, CD4, CK, CD8 and DAPI),
each with a manually annotated nuclear mask and a whole-cell mask ('<name>.tif', '<name>_GT_nuclei.tif' and
'<name>_GT_cells.tif'). The masks are instance labels, where the nucleus and the cell of an object share the same id.

The dataset is located at https://doi.org/10.5281/zenodo.22107836, released under a CC-BY-4.0 license.
Please cite the corresponding Zenodo record if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://zenodo.org/api/records/22107836/files"

CHECKSUMS = {
    "23B10981_1_5_GT_cells.tif": "e14a640cb0a9d0bfa05c6f4b8418a0dee498b372afdc0f557e37c9f22822a764",
    "23B10981_1_5_GT_nuclei.tif": "9473b9406afff196ac932da6692a73748fbb9c20f4681ac9708723ea6c725d4e",
    "23B10981_1_5.tif": "9f04251bd4d13210343fcc36b41d3ffb59b248897f80a2018e9568d81c4f050f",
    "23B10981_9_15_GT_cells.tif": "5a9c323c2a181e76318e9d2b07b91e815a7443b9b16318561e538d72858e5b95",
    "23B10981_9_15_GT_nuclei.tif": "06afc949650b5be542ded08a3e6daae5e3c39dc91db0c7d6f7059b6c00d53dfd",
    "23B10981_9_15.tif": "f68d759e7ef734f30077333380132fb0bf82ed7c92efb96c581c9b2e383a1f0f",
    "24B2274_43_47_GT_cells.tif": "83e8b0810221d5fa3e208d2f8a27408c93bf52120a746d7a3840dd62945d2584",
    "24B2274_43_47_GT_nuclei.tif": "8c0871b27dc5bb2a3ab1443cb12011f7ee5f3f7dffb7e1d67b07499501a41d5f",
    "24B2274_43_47.tif": "b9b3a0f37089819c7604294fe97904660ff5097454feb5d7bea5c3a1fde25abc",
    "24B2274_45_30_GT_cells.tif": "6de91c905770ac92c8b115a251657e42fe17ca647b7fb0572f5b3e12e8926cd4",
    "24B2274_45_30_GT_nuclei.tif": "4e43d9c4e27be42bac4b3ab8c29adff6d1d6879887fcccff5ce74e222b635ac2",
    "24B2274_45_30.tif": "9f98bdd7d158221ba84d5c5b3a3c22f6597ed2fe8aa18092c369f88ff5dcfb2a",
    "25B01044_35_29_GT_cells.tif": "b690869bb27af6168cec0cd829db0e92926928bce5a5f9df9eb4be04cb01c9c2",
    "25B01044_35_29_GT_nuclei.tif": "02ad8aac22fae9b176109f7f61d3fa7ac4b153a2a84cacace545e34edaf5a379",
    "25B01044_35_29.tif": "c5656ce061df8bcd47558407664a776f357609319ca8d999f4f13e434caa17c6",
    "Ctrl_14_21_GT_cells.tif": "133f913f5839d9dfe060de78ea808e5896e5db0cdf33301add7610a9ec100914",
    "Ctrl_14_21_GT_nuclei.tif": "4be1554128c115d9023b3a4ac712ae88c68256f8a53c946363d5da523123fd36",
    "Ctrl_14_21.tif": "461b1028c7ad184d49b246101ed6b1348e8952825f3aaf2da52bab568482c4e9",
    "Ctrl_16_43_GT_cells.tif": "2edf66e06146a6397549f638161eebe669806a1b7ce44784bfee8537a388d2fe",
    "Ctrl_16_43_GT_nuclei.tif": "f88769584ac56e56ad6576ee96496ef2e726efce0af370a3ee905b4c8488f721",
    "Ctrl_16_43.tif": "ef59047b5efc430e96c1aa4061e2a18e50b80e30e2b221fd611ca248c234d229",
    "Panel33C_17_11_GT_cells.tif": "22267ab8423b180aff01ff7977439be1c96351e1755c59a7d10aefad48ac4aa4",
    "Panel33C_17_11_GT_nuclei.tif": "04738d4e3f07bf1e60ad70815ce6813033d2344fc728ae3256e00ed047692620",
    "Panel33C_17_11.tif": "e7fe7ecdac0adc658dcf8fbbc87db7921f72b43511c32e0af182088e0b0e1163",
    "Panel33C_22_26_GT_cells.tif": "f8558e71ed10020009c864bb4c4d60a4731b41ef9c429d1d7c700f59863395d2",
    "Panel33C_22_26_GT_nuclei.tif": "ee51216b7ccbca2ba7a4754c10401e01948a242553aac896e2c1a199341d6520",
    "Panel33C_22_26.tif": "1013423eafb9e8e73db31e9678a33a8f569dbdd519e30272ab5eaca6c6b8df37",
    "Panel33C_26_7_GT_cells.tif": "769b9143234dfa1bd9958ffd550f91f974e4c02c2f1da41e96a54b06140500d5",
    "Panel33C_26_7_GT_nuclei.tif": "a9c5b89e3abcffa1ef6d1ad680ccbee22c8a0a5f0818be0021a2f5025eb14fd8",
    "Panel33C_26_7.tif": "5909843c673515b4a500f72fca96a60dc12183d4f89ec963833fd79bf19f6356",
}

LABEL_TYPES = ["nuclei", "cells"]


def get_mif_tonsil_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Multiplex IF Tonsil dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the downloaded data.
    """
    os.makedirs(path, exist_ok=True)
    for fname, checksum in CHECKSUMS.items():
        util.download_source(
            path=os.path.join(path, fname), url=f"{BASE_URL}/{fname}/content", download=download, checksum=checksum,
        )

    return path


def get_mif_tonsil_paths(
    path: Union[os.PathLike, str], label_type: Literal["nuclei", "cells"] = "nuclei", download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the Multiplex IF Tonsil data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        label_type: The choice of labels. Either 'nuclei' or 'cells'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if label_type not in LABEL_TYPES:
        raise ValueError(f"'{label_type}' is not a valid label type. Choose one of {LABEL_TYPES}.")

    data_dir = get_mif_tonsil_data(path, download)

    label_paths = natsorted(glob(os.path.join(data_dir, f"*_GT_{label_type}.tif")))
    raw_paths = [p.replace(f"_GT_{label_type}.tif", ".tif") for p in label_paths]

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    assert all(os.path.exists(p) for p in raw_paths)

    return raw_paths, label_paths


def get_mif_tonsil_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    label_type: Literal["nuclei", "cells"] = "nuclei",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Multiplex IF Tonsil dataset for nucleus and cell instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training. The images have a size of 100x100 pixels.
        label_type: The choice of labels. Either 'nuclei' or 'cells'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_mif_tonsil_paths(path, label_type, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        with_channels=True,
        is_seg_dataset=True,
        ndim=2,
        **kwargs
    )


def get_mif_tonsil_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    label_type: Literal["nuclei", "cells"] = "nuclei",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Multiplex IF Tonsil dataloader for nucleus and cell instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training. The images have a size of 100x100 pixels.
        label_type: The choice of labels. Either 'nuclei' or 'cells'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_mif_tonsil_dataset(path, patch_shape, label_type, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
