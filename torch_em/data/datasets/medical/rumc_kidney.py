"""The RUMC Kidney dataset contains annotations for kidney and kidney abnormality
segmentation in contrast-enhanced thorax-abdomen CT scans.

The scans were collected at the Radboud University Medical Center (RUMC), Nijmegen.

The label ids are - kidney: 1, abnormality: 2

NOTE: The abnormality class merges the five types the authors annotated (tumors, cysts, masses,
lesions and metastases) into one id, i.e. tumors cannot be told apart from cysts. Use
`torch_em.data.datasets.medical.kits` if you need labels that separate tumors from cysts.

The dataset is located at https://doi.org/10.5281/zenodo.8014290.
This dataset is from the publication https://doi.org/10.48550/arXiv.2309.03383.
Please cite it if you use this dataset for your research.
"""

import os
import json
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Literal

import numpy as np

from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/8014290/files/{}?download=1"

# Zenodo publishes md5 only, whereas 'util.download_source' verifies sha256, so we can only check
# the segmentations. The md5 of the image archives is listed in the record linked above.
CHECKSUMS = {
    "images1.zip": None,
    "images2.zip": None,
    "images3.zip": None,
    "images4.zip": None,
    "segmentations.zip": "7d711e306e52fe37216c126c3b282935f916db594193a2d1e3fdf13a2757c163",
}

IMAGE_DIRS = ("images1", "images2", "images3", "images4")
LABEL_IDS = {"kidney": 1, "abnormality": 2}
VALID_SPLITS = ("train", "val", "test")
METAIMAGE_EXTS = (".mha", ".mhd")

# The NIfTI extension code for free-form text, which we use to store the original header as json.
NIFTI_COMMENT_ECODE = 6


def _read_metaimage_header(path):
    header, is_mha = {}, path.endswith(".mha")
    with open(path, "rb") as f:
        for line in f:
            if b"=" not in line:
                break
            key, value = line.decode("latin-1").split("=", 1)
            key = key.strip()
            header[key] = value.strip()
            # For '.mha' the binary data directly follows the header, so we must not read any further.
            if is_mha and key == "ElementDataFile":
                break
    return header


def _itk_geometry_to_ras_affine(spacing, origin, direction):
    affine = np.eye(4, dtype="float64")
    direction = np.asarray(direction, dtype="float64").reshape(3, 3)
    affine[:3, :3] = direction @ np.diag(np.asarray(spacing, dtype="float64"))
    affine[:3, 3] = np.asarray(origin, dtype="float64")
    # ITK stores the geometry in LPS, whereas NIfTI expects RAS, so the first two axes flip sign.
    return np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine


def _convert_mha_to_nifti(path: str, output_path: str, keep_metadata: bool = True) -> str:
    """Convert a MetaImage file to the NIfTI format.

    The voxel spacing, origin and direction are mapped to the NIfTI affine, so that the converted
    volume keeps its physical geometry. The original MetaImage header is stored verbatim in a NIfTI
    header extension, so that the metadata stays inside the converted file.

    Requires the SimpleITK python library.

    Args:
        path: Filepath to the MetaImage file ('.mha' or '.mhd').
        output_path: Filepath for the converted file. Use a '.nii' suffix to write an uncompressed
            volume, which can be memory-mapped for lazy loading, or '.nii.gz' to write a compressed one.
        keep_metadata: Whether to store the original header in a NIfTI header extension.

    Returns:
        The filepath to the converted file.
    """
    import nibabel as nib
    import SimpleITK as sitk

    if not path.endswith(METAIMAGE_EXTS):
        raise ValueError(f"The provided file ({path}) isn't in MetaImage format.")

    image = sitk.ReadImage(path)
    affine = _itk_geometry_to_ras_affine(image.GetSpacing(), image.GetOrigin(), image.GetDirection())

    # SimpleITK returns the array in 'zyx' order, whereas the NIfTI affine refers to 'xyz'.
    data = sitk.GetArrayFromImage(image).transpose(2, 1, 0)

    nifti = nib.Nifti1Image(data, affine)
    nifti.set_data_dtype(data.dtype)
    nifti.header.set_xyzt_units(xyz="mm")

    if keep_metadata:
        metadata = {
            "metaimage_header": _read_metaimage_header(path),
            "itk_metadata": {k: image.GetMetaData(k) for k in image.GetMetaDataKeys()},
        }
        payload = json.dumps(metadata).encode("utf-8")
        nifti.header.extensions.append(nib.nifti1.Nifti1Extension(NIFTI_COMMENT_ECODE, payload))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    nib.save(nifti, output_path)
    return output_path


def _read_nifti_metadata(path: str) -> dict:
    """Read the metadata that `_convert_mha_to_nifti` stored in a NIfTI header extension.

    Args:
        path: Filepath to the NIfTI file.

    Returns:
        The original metadata. Empty if the file does not carry any.
    """
    import nibabel as nib

    for extension in nib.load(path).header.extensions:
        if extension.get_code() != NIFTI_COMMENT_ECODE:
            continue
        try:
            return json.loads(extension.get_content())
        except (ValueError, UnicodeDecodeError):  # Some other tool wrote a plain text comment.
            continue
    return {}


class _SelectLabel:
    def __init__(self, label_id):
        self.label_id = label_id

    def __call__(self, labels):
        return (labels == self.label_id).astype("uint8")


def _find_image_path(path, case_id):
    for image_dir in IMAGE_DIRS:
        image_path = os.path.join(path, image_dir, f"{case_id}.mha")
        if os.path.exists(image_path):
            return image_path

    raise FileNotFoundError(f"Could not find the image for case '{case_id}'.")


def _preprocess_inputs(path, compress):
    # The volumes are converted to nifti, because elf has no file wrapper for MetaImage and would
    # have to load each volume into memory as a whole.
    suffix = ".nii.gz" if compress else ".nii"

    image_dir = os.path.join(path, "preprocessed", "images")
    label_dir = os.path.join(path, "preprocessed", "labels")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    label_paths = natsorted(glob(os.path.join(path, "segmentations", "*.mha")))
    if not label_paths:
        raise RuntimeError(f"Could not find the segmentations in '{path}'.")

    for label_path in tqdm(label_paths, desc="Preprocessing inputs"):
        case_id = os.path.basename(label_path)[:-len("_segmentations.mha")]

        target_image_path = os.path.join(image_dir, f"{case_id}{suffix}")
        target_label_path = os.path.join(label_dir, f"{case_id}{suffix}")
        if os.path.exists(target_image_path) and os.path.exists(target_label_path):
            continue

        _convert_mha_to_nifti(_find_image_path(path, case_id), target_image_path)
        _convert_mha_to_nifti(label_path, target_label_path)


def get_rumc_kidney_data(
    path: Union[os.PathLike, str], download: bool = False, compress: bool = False
) -> str:
    """Download the RUMC Kidney dataset.

    The download is roughly 38 GB. The volumes are converted to nifti for lazy loading, which needs
    about 137 GB more when uncompressed. Set `compress` to trade disk space for loading speed:
    compressed volumes take up roughly 40 GB, but nifti cannot memory-map them, so sampling a random
    patch is more than an order of magnitude slower.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.
        compress: Whether to store the converted volumes as '.nii.gz' instead of '.nii'.

    Returns:
        The folder where the dataset is downloaded and preprocessed.
    """
    data_dir = os.path.join(path, "preprocessed")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    for fname, checksum in CHECKSUMS.items():
        zip_path = os.path.join(path, fname)
        util.download_source(path=zip_path, url=URL.format(fname), download=download, checksum=checksum)
        util.unzip(zip_path=zip_path, dst=path)

    _preprocess_inputs(path, compress)

    return data_dir


def _get_split_map(path, data_dir):
    split_path = os.path.join(path, "splits_rumc_kidney.json")
    if os.path.exists(split_path):
        with open(split_path) as f:
            return json.load(f)

    case_ids = [os.path.basename(p).split(".")[0] for p in glob(os.path.join(data_dir, "images", "*.nii*"))]
    train_ids, test_ids = train_test_split(natsorted(case_ids), test_size=0.25, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.1, random_state=42)

    split_map = {"train": train_ids, "val": val_ids, "test": test_ids}
    with open(split_path, "w") as f:
        json.dump(split_map, f, indent=2)

    return split_map


def get_rumc_kidney_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"],
    download: bool = False,
    compress: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the RUMC Kidney data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: Which data split to use.
        download: Whether to download the data if it is not present.
        compress: Whether the converted volumes are stored as '.nii.gz' instead of '.nii'.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"Invalid split '{split}'. Must be one of {VALID_SPLITS}.")

    data_dir = get_rumc_kidney_data(path, download, compress)
    split_map = _get_split_map(path, data_dir)

    suffix = ".nii.gz" if compress else ".nii"
    raw_paths = [os.path.join(data_dir, "images", f"{case_id}{suffix}") for case_id in split_map[split]]
    label_paths = [os.path.join(data_dir, "labels", f"{case_id}{suffix}") for case_id in split_map[split]]

    missing = [p for p in raw_paths + label_paths if not os.path.exists(p)]
    if missing:
        raise RuntimeError(f"Could not find {len(missing)} files, e.g. '{missing[0]}'.")

    return raw_paths, label_paths


def get_rumc_kidney_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal["train", "val", "test"],
    label_choice: Literal["all", "kidney", "abnormality"] = "all",
    resize_inputs: bool = False,
    download: bool = False,
    compress: bool = False,
    **kwargs
) -> Dataset:
    """Get the RUMC Kidney dataset for kidney and kidney abnormality segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: Which data split to use.
        label_choice: Which labels to use. 'all' keeps both classes, the other choices return a
            binary mask for that class alone.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        compress: Whether the converted volumes are stored as '.nii.gz' instead of '.nii'.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_rumc_kidney_paths(path, split, download, compress)

    if label_choice != "all":
        if label_choice not in LABEL_IDS:
            raise ValueError(f"Invalid label choice '{label_choice}'. Must be 'all' or one of {tuple(LABEL_IDS)}.")

        kwargs = util.update_kwargs(kwargs, "label_transform", _SelectLabel(LABEL_IDS[label_choice]))

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths, raw_key="data", label_paths=label_paths, label_key="data",
        patch_shape=patch_shape, **kwargs
    )


def get_rumc_kidney_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal["train", "val", "test"],
    label_choice: Literal["all", "kidney", "abnormality"] = "all",
    resize_inputs: bool = False,
    download: bool = False,
    compress: bool = False,
    **kwargs
) -> DataLoader:
    """Get the RUMC Kidney dataloader for kidney and kidney abnormality segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: Which data split to use.
        label_choice: Which labels to use. 'all' keeps both classes, the other choices return a
            binary mask for that class alone.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        compress: Whether the converted volumes are stored as '.nii.gz' instead of '.nii'.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_rumc_kidney_dataset(
        path, patch_shape, split, label_choice, resize_inputs, download, compress, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
