"""The SLiMIA dataset contains annotations for spheroid segmentation
in light microscopy images from 9 different microscopes and 47 cell lines.

NOTE: The annotations are semantic segmentation of spheroids.

The dataset provides images with binary manual segmentation masks of spheroids
formed using liquid overlay and hanging drop techniques.

The dataset is located at
https://figshare.com/collections/The_Spheroid_Light_Microscopy_Image_Atlas_SLiMIA_for_morphometrical_analysis_of_three_dimensional_cell_cultures/7486311.
This dataset is from the publication https://doi.org/10.1038/s41597-025-04441-x.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from typing import Union, Tuple, List, Optional

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "OperaPhenix": "https://ndownloader.figshare.com/files/50120850",
    "OlympusIX05": "https://ndownloader.figshare.com/files/50120853",
    "Axiovert200M": "https://ndownloader.figshare.com/files/50122224",
    "Cytation5": "https://ndownloader.figshare.com/files/50122194",
    "LeicaDMi3000B": "https://ndownloader.figshare.com/files/50122473",
    "Axiovert200": "https://ndownloader.figshare.com/files/50134212",
    "IncucyteS3": "https://ndownloader.figshare.com/files/50134218",
    "LeicaDMi1": "https://ndownloader.figshare.com/files/50134776",
    "IncucyteZOOM": "https://ndownloader.figshare.com/files/50136054",
}

MICROSCOPES = list(URLS.keys())


def _create_h5_data(path, microscope):
    """Create h5 files with raw images and binary spheroid labels."""
    import h5py
    import imageio.v3 as imageio
    from tqdm import tqdm

    h5_dir = os.path.join(path, "h5_data", microscope)
    os.makedirs(h5_dir, exist_ok=True)

    micro_dir = os.path.join(path, microscope)
    image_dir = os.path.join(micro_dir, "Images")
    seg_dir = os.path.join(micro_dir, "Manual segmentations")

    cell_lines = sorted(os.listdir(image_dir))

    for cell_line in cell_lines:
        cl_image_dir = os.path.join(image_dir, cell_line)
        cl_seg_dir = os.path.join(seg_dir, cell_line)

        if not os.path.isdir(cl_image_dir) or not os.path.isdir(cl_seg_dir):
            continue

        image_paths = sorted(glob(os.path.join(cl_image_dir, "*.tiff")))

        for image_path in tqdm(image_paths, desc=f"Creating h5 for {microscope}/{cell_line}"):
            # Match image to mask: image is .ome.tiff, mask is .tiff with the same base name.
            base_name = os.path.basename(image_path).replace(".ome.tiff", "").replace(".tiff", "")
            h5_path = os.path.join(h5_dir, f"{base_name}.h5")

            if os.path.exists(h5_path):
                continue

            # Try both naming conventions for the mask.
            seg_path = os.path.join(cl_seg_dir, f"{base_name}.tiff")
            if not os.path.exists(seg_path):
                seg_path = os.path.join(cl_seg_dir, f"{base_name}.ome.tiff")

            if not os.path.exists(seg_path):
                continue

            raw = imageio.imread(image_path)
            seg = imageio.imread(seg_path)

            # Convert binary mask (0/255) to labels (0/1).
            labels = (seg > 0).astype("int64")

            with h5py.File(h5_path, "w") as f:
                f.create_dataset("raw", data=raw, compression="gzip")
                f.create_dataset("labels", data=labels, compression="gzip")

    return h5_dir


def get_slimia_data(
    path: Union[os.PathLike, str],
    microscope: Optional[Union[str, List[str]]] = None,
    download: bool = False,
) -> str:
    """Download the SLiMIA dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        microscope: The microscope(s) to download data for. If None, all microscopes will be downloaded.
            Available microscopes: OperaPhenix, OlympusIX05, Axiovert200M, Cytation5,
            LeicaDMi3000B, Axiovert200, IncucyteS3, LeicaDMi1, IncucyteZOOM.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the directory with the data.
    """
    if microscope is None:
        microscope = MICROSCOPES
    elif isinstance(microscope, str):
        microscope = [microscope]

    for micro in microscope:
        assert micro in MICROSCOPES, f"'{micro}' is not a valid microscope. Choose from {MICROSCOPES}."

        micro_dir = os.path.join(path, micro)
        if os.path.exists(micro_dir):
            continue

        os.makedirs(path, exist_ok=True)
        zip_path = os.path.join(path, f"{micro}.zip")
        util.download_source(path=zip_path, url=URLS[micro], download=download, checksum=None)
        util.unzip(zip_path=zip_path, dst=path)

    return path


def get_slimia_paths(
    path: Union[os.PathLike, str],
    microscope: Optional[Union[str, List[str]]] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the SLiMIA data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        microscope: The microscope(s) to use. If None, all microscopes will be used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the h5 data.
    """
    from natsort import natsorted

    if microscope is None:
        microscope = MICROSCOPES
    elif isinstance(microscope, str):
        microscope = [microscope]

    get_slimia_data(path, microscope, download)

    all_h5_paths = []
    for micro in microscope:
        h5_dir = os.path.join(path, "h5_data", micro)
        if not os.path.exists(h5_dir) or len(glob(os.path.join(h5_dir, "*.h5"))) == 0:
            _create_h5_data(path, micro)

        h5_paths = glob(os.path.join(h5_dir, "*.h5"))
        all_h5_paths.extend(h5_paths)

    assert len(all_h5_paths) > 0, f"No data found for microscope(s) '{microscope}'"

    return natsorted(all_h5_paths)


def get_slimia_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    microscope: Optional[Union[str, List[str]]] = None,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the SLiMIA dataset for spheroid segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        microscope: The microscope(s) to use. If None, all microscopes will be used.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    h5_paths = get_slimia_paths(path, microscope, download)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True,
    )
    kwargs = util.ensure_transforms(ndim=2, **kwargs)

    return torch_em.default_segmentation_dataset(
        raw_paths=h5_paths,
        raw_key="raw",
        label_paths=h5_paths,
        label_key="labels",
        patch_shape=patch_shape,
        ndim=2,
        **kwargs
    )


def get_slimia_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    microscope: Optional[Union[str, List[str]]] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the SLiMIA dataloader for spheroid segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        microscope: The microscope(s) to use. If None, all microscopes will be used.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_slimia_dataset(
        path=path,
        patch_shape=patch_shape,
        microscope=microscope,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
