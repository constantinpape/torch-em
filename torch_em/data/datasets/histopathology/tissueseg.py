"""This dataset contains annotations for tissue vs. background segmentation in
whole-slide histopathology images of multiple organs and stains.

The dataset is a representative sample from the publication
https://doi.org/10.7717/peerj.8242 ("Resolution-agnostic tissue segmentation
in whole-slide histopathology images with convolutional neural networks") and
is hosted on Zenodo at https://doi.org/10.5281/zenodo.3375528.
Please cite it if you use this dataset in your research.

The data consists of ten whole-slide images. The five 'development' samples
(breast, breast lymph node and three tongue stains) provide annotations at a
pixel spacing of 0.5 micrometer, with both a binary (tissue vs. background) and
a six-class label. The five 'dissimilar' samples (brain, cornea, kidney, skin
and uterus) provide only the binary annotation at a pixel spacing of 2.0
micrometer.

The label values follow the scheme used by the authors. The unannotated regions
(typically the glass background outside the region of interest) are labeled as 0.
The binary annotations use 3 for non-tissue and 6 for tissue. The six-class
annotations additionally use 1 (edge artifacts), 2 (inner artifacts),
4 (external margin) and 5 (internal margin), with 3 for background and 6 for tissue.

NOTE: The whole-slide images and masks are multi-resolution pyramidal TIFFs of
several gigabytes each. On the first use each requested sample is converted into
a chunked HDF5 file at the resolution matching its annotation, which requires
some time and disk space.
"""

import os
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

from tqdm import tqdm

import torch

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://zenodo.org/api/records/3375528/files"

SAMPLES = {
    "breast": "breast_hne_00",
    "breast_lymph_node": "breast_lymph_node_hne_00",
    "tongue_hne": "tongue_hne_00",
    "tongue_ki67": "tongue_ki67_00",
    "tongue_ae1ae3": "tongue_ae1ae3_00",
    "brain": "brain_alcianblue_00",
    "cornea": "cornea_grocott_00",
    "kidney": "kidney_cab_00",
    "skin": "skin_perls_00",
    "uterus": "uterus_vonkossa_00",
}

# The development samples are densely annotated at 0.5 micrometer with binary and six-class labels.
DEVELOPMENT_SAMPLES = ["breast", "breast_lymph_node", "tongue_hne", "tongue_ki67", "tongue_ae1ae3"]

# The dissimilar samples only have a binary annotation at 2.0 micrometer.
DISSIMILAR_SAMPLES = ["brain", "cornea", "kidney", "skin", "uterus"]


def _mask_filename(sample, annotations):
    stem = SAMPLES[sample]
    if sample in DEVELOPMENT_SAMPLES:
        return f"{stem}_mask_cl2_sp0.5.tif" if annotations == "binary" else f"{stem}_mask_cl6_sp0.5.tif"
    else:
        return f"{stem}_mask_cl2_sp2.0.tif"


def _resolve_samples(samples, annotations):
    if annotations not in ("binary", "semantic"):
        raise ValueError(f"'{annotations}' is not a valid annotation choice. Use 'binary' or 'semantic'.")

    if samples is None:
        samples = DEVELOPMENT_SAMPLES if annotations == "semantic" else list(SAMPLES.keys())

    if isinstance(samples, str):
        samples = [samples]

    for sample in samples:
        if sample not in SAMPLES:
            raise ValueError(f"'{sample}' is not a valid sample. Choose from {list(SAMPLES.keys())}.")
        if annotations == "semantic" and sample not in DEVELOPMENT_SAMPLES:
            raise ValueError(f"The sample '{sample}' does not have semantic annotations. Use annotations='binary'.")

    return samples


def _download_file(path, filename, download):
    out_path = os.path.join(path, filename)
    if os.path.exists(out_path):
        return out_path

    # The whole-slide images are several gigabytes each, so we do not verify checksums.
    url = f"{BASE_URL}/{filename}/content"
    util.download_source(path=out_path, url=url, download=download, checksum=None)
    return out_path


def _open_level(series, level_index):
    import zarr

    # The pyramidal TIFFs are natively tiled, so a zarr view reads only the requested tiles lazily.
    # Multi-level series open as a group keyed by the level index; single-level series open as an array.
    array = zarr.open(series.aszarr(), mode="r")
    return array if hasattr(array, "shape") else array[str(level_index)]


def _convert_sample(wsi_path, mask_paths, output_path, tile=4096):
    import h5py
    import tifffile

    # All annotations of a sample share the same resolution, so the raw level is matched to the first mask.
    first_mask = tifffile.TiffFile(list(mask_paths.values())[0])
    height, width = first_mask.series[0].levels[0].shape

    image_series = tifffile.TiffFile(wsi_path).series[0]
    level_index = next((i for i, lv in enumerate(image_series.levels) if lv.shape[:2] == (height, width)), None)
    if level_index is None:
        raise RuntimeError(
            f"Could not find a resolution level in '{wsi_path}' matching the mask shape ({height}, {width})."
        )

    image = _open_level(image_series, level_index)
    masks = {ann: _open_level(tifffile.TiffFile(p).series[0], 0) for ann, p in mask_paths.items()}
    for ann, mask in masks.items():
        if mask.shape != (height, width):
            raise RuntimeError(f"Mask '{ann}' shape {mask.shape} does not match the raw shape ({height}, {width}).")

    tmp_path = output_path + ".tmp"
    with h5py.File(tmp_path, "w") as f:
        raw = f.create_dataset(
            "images/raw", shape=(3, height, width), dtype="uint8", compression="gzip", chunks=(1, 512, 512)
        )
        mask_datasets = {}
        for ann in masks:
            mask_datasets[ann] = f.create_dataset(
                f"labels/{ann}", shape=(height, width), dtype="uint8", compression="gzip", chunks=(512, 512)
            )
        for y in tqdm(range(0, height, tile), desc=f"Converting {Path(wsi_path).stem}"):
            for x in range(0, width, tile):
                th, tw = min(tile, height - y), min(tile, width - x)
                raw[:, y:y + th, x:x + tw] = image[y:y + th, x:x + tw].transpose(2, 0, 1)
                for ann in masks:
                    mask_datasets[ann][y:y + th, x:x + tw] = masks[ann][y:y + th, x:x + tw]

    os.replace(tmp_path, output_path)


def get_tissueseg_data(
    path: Union[os.PathLike, str],
    samples: Optional[Union[str, List[str]]] = None,
    annotations: Literal["binary", "semantic"] = "binary",
    download: bool = False,
) -> str:
    """Download and preprocess the tissue segmentation data.

    Args:
        path: Filepath to a folder where the data will be saved.
        samples: The samples to use. By default all samples valid for the annotation choice are used.
        annotations: The annotation type. Either 'binary' (tissue vs. background) or 'semantic' (six classes).
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder where the preprocessed data is stored.
    """
    samples = _resolve_samples(samples, annotations)

    raw_dir = os.path.join(path, "raw")
    preprocessed_dir = os.path.join(path, "preprocessed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(preprocessed_dir, exist_ok=True)

    for sample in samples:
        output_path = os.path.join(preprocessed_dir, f"{sample}.h5")
        if os.path.exists(output_path):
            continue

        wsi_path = _download_file(raw_dir, f"{SAMPLES[sample]}.tif", download)

        # Store all available masks for the sample so that 'binary' and 'semantic' share one file.
        mask_choices = ["binary", "semantic"] if sample in DEVELOPMENT_SAMPLES else ["binary"]
        mask_paths = {
            choice: _download_file(raw_dir, _mask_filename(sample, choice), download) for choice in mask_choices
        }

        _convert_sample(wsi_path, mask_paths, output_path)

    return preprocessed_dir


def get_tissueseg_paths(
    path: Union[os.PathLike, str],
    samples: Optional[Union[str, List[str]]] = None,
    annotations: Literal["binary", "semantic"] = "binary",
    download: bool = False,
) -> List[str]:
    """Get paths to the tissue segmentation data.

    Args:
        path: Filepath to a folder where the data will be saved.
        samples: The samples to use. By default all samples valid for the annotation choice are used.
        annotations: The annotation type. Either 'binary' (tissue vs. background) or 'semantic' (six classes).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the preprocessed HDF5 files.
    """
    samples = _resolve_samples(samples, annotations)
    preprocessed_dir = get_tissueseg_data(path, samples, annotations, download)
    volume_paths = [os.path.join(preprocessed_dir, f"{sample}.h5") for sample in samples]

    missing = [p for p in volume_paths if not os.path.exists(p)]
    if missing:
        raise RuntimeError(f"Could not find the preprocessed data at {missing}.")

    return volume_paths


def get_tissueseg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    samples: Optional[Union[str, List[str]]] = None,
    annotations: Literal["binary", "semantic"] = "binary",
    download: bool = False,
    label_dtype: torch.dtype = torch.int64,
    resize_inputs: bool = False,
    **kwargs
) -> Dataset:
    """Get the tissue segmentation dataset for tissue vs. background segmentation in whole-slide images.

    Args:
        path: Filepath to a folder where the data will be saved.
        patch_shape: The patch shape to use for training.
        samples: The samples to use. By default all samples valid for the annotation choice are used.
        annotations: The annotation type. Either 'binary' (tissue vs. background) or 'semantic' (six classes).
        download: Whether to download the data if it is not present.
        label_dtype: The datatype of the labels.
        resize_inputs: Whether to resize the input images.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_tissueseg_paths(path, samples, annotations, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="images/raw",
        label_paths=volume_paths,
        label_key=f"labels/{annotations}",
        patch_shape=patch_shape,
        label_dtype=label_dtype,
        is_seg_dataset=True,
        with_channels=True,
        ndim=2,
        **kwargs
    )


def get_tissueseg_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    samples: Optional[Union[str, List[str]]] = None,
    annotations: Literal["binary", "semantic"] = "binary",
    download: bool = False,
    label_dtype: torch.dtype = torch.int64,
    resize_inputs: bool = False,
    **kwargs
) -> DataLoader:
    """Get the tissue segmentation dataloader for tissue vs. background segmentation in whole-slide images.

    Args:
        path: Filepath to a folder where the data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        samples: The samples to use. By default all samples valid for the annotation choice are used.
        annotations: The annotation type. Either 'binary' (tissue vs. background) or 'semantic' (six classes).
        download: Whether to download the data if it is not present.
        label_dtype: The datatype of the labels.
        resize_inputs: Whether to resize the input images.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_tissueseg_dataset(
        path=path, patch_shape=patch_shape, samples=samples, annotations=annotations, download=download,
        label_dtype=label_dtype, resize_inputs=resize_inputs, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
