"""The MUCIC (Masaryk University Cell Image Collection) datasets contain synthetic 3D
microscopy images for cell and nucleus segmentation benchmarking.

NOTE: Most of the datasets available at MUCIC are synthetic images (see detailed description below).

Available datasets:
- Colon Tissue: 30 synthetic 3D images of human colon tissue with semantic segmentation labels
- HL60 Cell Line: Synthetic 3D images of HL60 cells with instance segmentation labels
- Granulocytes: Synthetic 3D images of granulocytes with instance segmentation labels
- Vasculogenesis: Time-lapse 2D images of living cells with semantic segmentation labels

The datasets are from CBIA (Centre for Biomedical Image Analysis) at Masaryk University.

The data is located at https://cbia.fi.muni.cz/datasets/.
- Colon Tissue: https://doi.org/10.1007/978-3-642-21593-3_4
- HL60 Cell Line: https://doi.org/10.1002/cyto.a.20811
- Granulocytes: https://doi.org/10.1002/cyto.a.20811
- Vasculogenesis: https://doi.org/10.1109/ICIP.2016.7532871
Please cite the relevant publication if you use this dataset in your research.
"""

import os
from glob import glob
from typing import Union, Tuple, List, Optional

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "colon_tissue": {
        "low": "https://datasets.gryf.fi.muni.cz/iciar2011/ColonTissue_LowNoise_3D_HDF5.zip",
        "high": "https://datasets.gryf.fi.muni.cz/iciar2011/ColonTissue_HighNoise_3D_HDF5.zip",
    },
    "hl60": {
        "low_c00": "https://datasets.gryf.fi.muni.cz/cytometry2009/HL60_LowNoise_C00_3D_HDF5.zip",
        "low_c25": "https://datasets.gryf.fi.muni.cz/cytometry2009/HL60_LowNoise_C25_3D_HDF5.zip",
        "low_c50": "https://datasets.gryf.fi.muni.cz/cytometry2009/HL60_LowNoise_C50_3D_HDF5.zip",
        "low_c75": "https://datasets.gryf.fi.muni.cz/cytometry2009/HL60_LowNoise_C75_3D_HDF5.zip",
        "high_c00": "https://datasets.gryf.fi.muni.cz/cytometry2009/HL60_HighNoise_C00_3D_HDF5.zip",
        "high_c25": "https://datasets.gryf.fi.muni.cz/cytometry2009/HL60_HighNoise_C25_3D_HDF5.zip",
        "high_c50": "https://datasets.gryf.fi.muni.cz/cytometry2009/HL60_HighNoise_C50_3D_HDF5.zip",
        "high_c75": "https://datasets.gryf.fi.muni.cz/cytometry2009/HL60_HighNoise_C75_3D_HDF5.zip",
    },
    "granulocytes": {
        "low": "https://datasets.gryf.fi.muni.cz/cytometry2009/Granulocytes_LowNoise_3D_HDF5.zip",
        "high": "https://datasets.gryf.fi.muni.cz/cytometry2009/Granulocytes_HighNoise_3D_HDF5.zip",
    },
    "vasculogenesis": {
        "default": {
            "images": "https://datasets.gryf.fi.muni.cz/icip2016/vasculogenesis-images.zip",
            "labels": "https://datasets.gryf.fi.muni.cz/icip2016/vasculogenesis-labels.zip",
        },
    },
}

CELL_LINES = list(URLS.keys())


def _get_variants(cell_line):
    return list(URLS[cell_line].keys())


# Cell lines with semantic labels that need connected components for instance segmentation
_SEMANTIC_LABEL_CELL_LINES = ["colon_tissue", "vasculogenesis"]

# Cell lines with separate image/label zip files
_SEPARATE_ZIPS_CELL_LINES = ["vasculogenesis"]


def _create_mucic_h5(path, cell_line, variant):
    """Create processed h5 files from raw and label files."""
    import h5py
    from tqdm import tqdm

    data_dir = os.path.join(path, cell_line, variant)
    h5_out_dir = os.path.join(path, cell_line, "processed", variant)
    os.makedirs(h5_out_dir, exist_ok=True)

    # Find all raw files (image-final_*.h5)
    raw_files = sorted(glob(os.path.join(data_dir, "**", "image-final_*.h5"), recursive=True))
    if not raw_files:
        raw_files = sorted(glob(os.path.join(data_dir, "**", "image-final_*.hdf5"), recursive=True))

    needs_connected_components = cell_line in _SEMANTIC_LABEL_CELL_LINES

    for raw_path in tqdm(raw_files, desc=f"Processing {cell_line} {variant} data"):
        # Find corresponding label file
        label_path = raw_path.replace("image-final_", "image-labels_")
        if not os.path.exists(label_path):
            continue

        # Get output filename
        fname = os.path.basename(raw_path)
        out_fname = fname.replace("image-final_", f"{cell_line}_").replace(".hdf5", ".h5")
        out_path = os.path.join(h5_out_dir, out_fname)

        if os.path.exists(out_path):
            continue

        with h5py.File(raw_path, "r") as f:
            raw = f["Image"][:]

        with h5py.File(label_path, "r") as f:
            labels = f["Image"][:]

        # Convert semantic labels to instance labels if needed
        if needs_connected_components:
            from skimage.measure import label
            instances = label(labels > 0).astype("int64")
        else:
            instances = labels.astype("int64")

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels/instances", data=instances, compression="gzip")
            f.create_dataset("labels/semantic", data=(labels > 0).astype("uint8"), compression="gzip")

    return h5_out_dir


def _semantic_to_instances_watershed(semantic_mask, erosion_iterations=2):
    """Convert semantic mask to instance labels using erosion + watershed.

    This handles cases where cells are touching by a few pixels:
    1. Erode the mask to separate touching cells
    2. Run connected components on eroded mask to get seed labels
    3. Use watershed to expand seeds back to original mask boundaries
    """
    from scipy.ndimage import binary_erosion, distance_transform_edt
    from skimage.measure import label
    from skimage.segmentation import watershed

    binary_mask = semantic_mask > 0

    # Erode to separate touching cells
    eroded = binary_erosion(binary_mask, iterations=erosion_iterations)

    # Get seed labels from eroded mask
    seeds = label(eroded)

    # Use watershed to expand seeds to fill original mask
    # Distance transform gives us the "landscape" for watershed
    distance = distance_transform_edt(binary_mask)
    instances = watershed(-distance, seeds, mask=binary_mask)

    return instances.astype("int64")


def _create_vasculogenesis_h5(path, variant):
    """Create processed h5 files for vasculogenesis from separate image/label PNG directories."""
    import h5py
    import imageio.v2 as imageio
    from tqdm import tqdm

    data_dir = os.path.join(path, "vasculogenesis", variant)
    h5_out_dir = os.path.join(path, "vasculogenesis", "processed", variant)
    os.makedirs(h5_out_dir, exist_ok=True)

    # Find image and label directories
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")

    # Find all PNG image files (image_XXXX.png)
    raw_files = sorted(glob(os.path.join(images_dir, "*.png")))

    for raw_path in tqdm(raw_files, desc=f"Processing vasculogenesis {variant} data"):
        # Find corresponding label file (image_XXXX.png -> mask_XXXX.png)
        fname = os.path.basename(raw_path)
        label_fname = fname.replace("image_", "mask_")
        label_path = os.path.join(labels_dir, label_fname)

        if not os.path.exists(label_path):
            continue

        # Output filename
        file_id = fname.replace("image_", "").replace(".png", "")
        out_fname = f"vasculogenesis_{file_id}.h5"
        out_path = os.path.join(h5_out_dir, out_fname)

        if os.path.exists(out_path):
            continue

        raw = imageio.imread(raw_path)
        labels_data = imageio.imread(label_path)

        # Convert semantic labels to instance labels using erosion + watershed
        instances = _semantic_to_instances_watershed(labels_data)

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels/instances", data=instances, compression="gzip")
            f.create_dataset("labels/semantic", data=(labels_data > 0).astype("uint8"), compression="gzip")

    return h5_out_dir


def get_mucic_data(
    path: Union[os.PathLike, str],
    cell_line: str,
    variant: Optional[Union[str, List[str]]] = None,
    download: bool = False,
) -> str:
    """Download the MUCIC dataset for a specific cell line.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        cell_line: The cell line to use. One of 'colon_tissue', 'hl60', 'granulocytes', or 'vasculogenesis'.
        variant: The dataset variant(s).
            For 'colon_tissue' and 'granulocytes': 'low' or 'high' (noise levels).
            For 'hl60': combination of noise ('low', 'high') and clustering ('c00', 'c25', 'c50', 'c75'),
            e.g. 'low_c00'.
            For 'vasculogenesis': 'default'.
            If None, downloads all variants for the selected cell line.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the dataset directory.
    """
    assert cell_line in CELL_LINES, f"'{cell_line}' is not valid. Choose from {CELL_LINES}."

    valid_variants = _get_variants(cell_line)
    if variant is None:
        variant = valid_variants
    elif isinstance(variant, str):
        variant = [variant]

    for v in variant:
        assert v in valid_variants, f"'{v}' is not valid for '{cell_line}'. Choose from {valid_variants}."

        data_dir = os.path.join(path, cell_line, v)
        if os.path.exists(data_dir) and len(glob(os.path.join(data_dir, "**", "*.h5"), recursive=True)) > 0:
            continue

        os.makedirs(data_dir, exist_ok=True)

        # Handle cell lines with separate image/label zip files
        if cell_line in _SEPARATE_ZIPS_CELL_LINES:
            urls = URLS[cell_line][v]
            # Download and extract images
            images_zip = os.path.join(path, f"{cell_line}_{v}_images.zip")
            util.download_source(path=images_zip, url=urls["images"], download=download, checksum=None)
            util.unzip(zip_path=images_zip, dst=os.path.join(data_dir, "images"), remove=False)
            # Download and extract labels
            labels_zip = os.path.join(path, f"{cell_line}_{v}_labels.zip")
            util.download_source(path=labels_zip, url=urls["labels"], download=download, checksum=None)
            util.unzip(zip_path=labels_zip, dst=os.path.join(data_dir, "labels"), remove=False)
        else:
            zip_path = os.path.join(path, f"{cell_line}_{v}.zip")
            util.download_source(path=zip_path, url=URLS[cell_line][v], download=download, checksum=None)
            util.unzip(zip_path=zip_path, dst=data_dir, remove=False)

    return os.path.join(path, cell_line)


def get_mucic_paths(
    path: Union[os.PathLike, str],
    cell_line: str,
    variant: Optional[Union[str, List[str]]] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the MUCIC data for a specific cell line.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        cell_line: The cell line to use. One of 'colon_tissue', 'hl60', 'granulocytes', or 'vasculogenesis'.
        variant: The dataset variant(s). If None, uses all variants.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the processed h5 data.
    """
    from natsort import natsorted

    assert cell_line in CELL_LINES, f"'{cell_line}' is not valid. Choose from {CELL_LINES}."

    get_mucic_data(path, cell_line, variant, download)

    valid_variants = _get_variants(cell_line)
    if variant is None:
        variant = valid_variants
    elif isinstance(variant, str):
        variant = [variant]

    all_h5_paths = []
    for v in variant:
        h5_out_dir = os.path.join(path, cell_line, "processed", v)

        # Process data if not already done
        if not os.path.exists(h5_out_dir) or len(glob(os.path.join(h5_out_dir, "*.h5"))) == 0:
            if cell_line == "vasculogenesis":
                _create_vasculogenesis_h5(path, v)
            else:
                _create_mucic_h5(path, cell_line, v)

        h5_paths = glob(os.path.join(h5_out_dir, "*.h5"))
        all_h5_paths.extend(h5_paths)

    assert len(all_h5_paths) > 0, f"No data found for cell_line '{cell_line}', variant '{variant}'"

    return natsorted(all_h5_paths)


def get_mucic_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    cell_line: str,
    variant: Optional[Union[str, List[str]]] = None,
    segmentation_type: str = "instances",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the MUCIC dataset for 3D cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        cell_line: The cell line to use. One of 'colon_tissue', 'hl60', 'granulocytes', or 'vasculogenesis'.
        variant: The dataset variant(s).
            For 'colon_tissue' and 'granulocytes': 'low' or 'high' (noise levels).
            For 'hl60': combination of noise ('low', 'high') and clustering ('c00', 'c25', 'c50', 'c75'),
            e.g. 'low_c00'.
            For 'vasculogenesis': 'default'.
            If None, uses all variants for the selected cell line.
        segmentation_type: The type of segmentation labels to use.
            One of 'instances' or 'semantic' (binary mask).
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert segmentation_type in ("instances", "semantic"), \
        f"'{segmentation_type}' is not valid. Choose from 'instances' or 'semantic'."

    h5_paths = get_mucic_paths(path, cell_line, variant, download)

    label_key = f"labels/{segmentation_type}"

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, label_dtype=np.int64,
    )

    # Vasculogenesis is 2D data, others are 3D
    ndim = 2 if cell_line == "vasculogenesis" else 3

    return torch_em.default_segmentation_dataset(
        raw_paths=h5_paths,
        raw_key="raw",
        label_paths=h5_paths,
        label_key=label_key,
        patch_shape=patch_shape,
        ndim=ndim,
        **kwargs
    )


def get_mucic_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    cell_line: str,
    variant: Optional[Union[str, List[str]]] = None,
    segmentation_type: str = "instances",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the MUCIC dataloader for 3D cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        cell_line: The cell line to use. One of 'colon_tissue', 'hl60', 'granulocytes', or 'vasculogenesis'.
        variant: The dataset variant(s).
            For 'colon_tissue' and 'granulocytes': 'low' or 'high' (noise levels).
            For 'hl60': combination of noise ('low', 'high') and clustering ('c00', 'c25', 'c50', 'c75'),
            e.g. 'low_c00'.
            For 'vasculogenesis': 'default'.
            If None, uses all variants for the selected cell line.
        segmentation_type: The type of segmentation labels to use.
            One of 'instances' or 'semantic' (binary mask).
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_mucic_dataset(
        path=path,
        patch_shape=patch_shape,
        cell_line=cell_line,
        variant=variant,
        segmentation_type=segmentation_type,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
