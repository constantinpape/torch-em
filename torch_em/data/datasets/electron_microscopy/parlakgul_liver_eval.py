"""The Parlakgul liver evaluation dataset contains manually annotated FIB-SEM crops of mouse liver
for mitochondria and lipid droplets. They are the independent evaluation volumes of the model-generated
segmentations of the Parlakgul liver dataset (see `parlakgul_liver.py`).

For each of the four liver samples (6461, 6464, 9430, 1857) there are three crops of 25 x 150 x 150 voxels
(z, y, x) per organelle, at 8 nm isotropic resolution. Every crop comes with:
- the raw image,
- the manual annotation (instance ids, 0 is background; for a few crops the annotation is semantic),
- the instance segmentation predicted by the authors' model for the same crop (ids are arbitrary).

This dataset is from the publication https://doi.org/10.1038/s41586-022-04518-2.
Please cite it if you use this dataset in your research.

The data is publicly available at https://www.ebi.ac.uk/empiar/EMPIAR-10791/.
"""

import io
import os
import re
import shutil
import zipfile
from typing import List, Literal, Tuple, Union

import numpy as np
from torch.utils.data import DataLoader, Dataset

import torch_em
from .. import util


URL = (
    "https://ftp.ebi.ac.uk/empiar/world_availability/10791/data/"
    "Parlakgul%20et%20al%20-%20Regulation%20of%20liver%20subcellular%20architecture%20controls%20metabolic%20"
    "homeostasis/FIB-SEM%20Raw%20and%20Segmentation%20Data/Segmentation%20Evaluation/"
    "Manual%20annotation%20-%20Prediction%20-%20Raw%20Slices.zip"
)
CHECKSUM = "bfb94ec5818a5b25e7ae380e624a29df366e32a66a674ad71030312951c9775f"
ZIP_NAME = "Manual annotation - Prediction - Raw Slices.zip"

SAMPLES = ("6461", "6464", "9430", "1857")
LABEL_CHOICES = {"mito": "mitochondria", "lipid_droplet": "lipid droplets"}
N_EVALUATION_VOLUMES = 3


def _read_tiff_stack(archive, name):
    """Read all slices of a nested zip archive as a (z, y, x) array, ordered by the slice number."""
    from PIL import Image

    with zipfile.ZipFile(io.BytesIO(archive.read(name))) as zf:
        members = [
            info.filename for info in zf.infolist()
            if info.filename.lower().endswith((".tif", ".tiff")) and "__MACOSX" not in info.filename
        ]
        members.sort(key=lambda f: int(re.search(r"(\d+)\.tiff?$", f.lower()).group(1)))
        return np.stack([np.array(Image.open(io.BytesIO(zf.read(f)))) for f in members])


def _find_member(names, sample, organelle, index, kind):
    # The archives are not named consistently: the manual annotation of 6464 is called 'manual evaluation'.
    pattern = {
        "raw": r"raw", "labels": r"manual (annotation|evaluation)", "prediction": r"prediction",
    }[kind]
    regex = re.compile(rf"^{sample}[ _].*{organelle} evaluation 00{index} {pattern}\.zip$")
    matches = [n for n in names if regex.match(os.path.basename(n))]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one archive for {sample}, {organelle}, {index}, {kind}, found {matches}.")
    return matches[0]


def get_parlakgul_liver_eval_data(
    path: Union[os.PathLike, str],
    sample: Literal["6461", "6464", "9430", "1857"] = "6461",
    label_choice: Literal["mito", "lipid_droplet"] = "mito",
    download: bool = False,
) -> List[str]:
    """Download the Parlakgul liver evaluation crops and convert them to zarr stores.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        sample: Which liver sample to use. One of "6461" (lean), "6464" (obese 1),
            "9430" (obese 2), "1857" (obese Climp63).
        label_choice: The organelle. One of "mito" or "lipid_droplet".
        download: Whether to download the data if it is not present.

    Returns:
        The filepaths to the zarr stores of the three evaluation crops.
    """
    import zarr
    from zarr.codecs import BloscCodec

    if sample not in SAMPLES:
        raise ValueError(f"'{sample}' is not a valid sample. Choose from {SAMPLES}.")
    if label_choice not in LABEL_CHOICES:
        raise ValueError(f"'{label_choice}' is not a valid label_choice. Choose from {list(LABEL_CHOICES)}.")
    organelle = LABEL_CHOICES[label_choice]

    os.makedirs(path, exist_ok=True)
    zarr_paths = [
        os.path.join(path, f"{sample}_{label_choice}_{i:03d}.zarr") for i in range(1, N_EVALUATION_VOLUMES + 1)
    ]
    if all(os.path.exists(p) for p in zarr_paths):
        return zarr_paths

    zip_path = os.path.join(path, ZIP_NAME)
    util.download_source(zip_path, URL, download, checksum=CHECKSUM)

    with zipfile.ZipFile(zip_path) as archive:
        names = [n for n in archive.namelist() if n.endswith(".zip") and "__MACOSX" not in n]
        for index, zarr_path in enumerate(zarr_paths, start=1):
            if os.path.exists(zarr_path):
                continue

            stacks = {
                kind: _read_tiff_stack(archive, _find_member(names, sample, organelle, index, kind))
                for kind in ("raw", "labels", "prediction")
            }
            # One crop (6461 mito 003) is annotated with 255 instead of 1.
            if stacks["labels"].max() == 255:
                stacks["labels"] = (stacks["labels"] > 0).astype("uint8")

            tmp_path = f"{zarr_path}.incomplete"
            if os.path.exists(tmp_path):
                shutil.rmtree(tmp_path)
            root = zarr.open_group(tmp_path, mode="w")
            root.attrs.update({"sample": sample, "label_choice": label_choice, "resolution_nm": [8, 8, 8]})
            for name, data in stacks.items():
                arr = root.create_array(
                    name, shape=data.shape, chunks=data.shape, dtype=data.dtype,
                    compressors=BloscCodec(cname="zstd", clevel=6),
                )
                arr[:] = data
            os.rename(tmp_path, zarr_path)

    return zarr_paths


def get_parlakgul_liver_eval_paths(
    path: Union[os.PathLike, str],
    samples: Union[str, List[str]] = "6461",
    label_choice: Literal["mito", "lipid_droplet"] = "mito",
    download: bool = False,
) -> List[str]:
    """Get paths to the Parlakgul liver evaluation zarr stores.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        samples: The liver sample(s) to use. One or several of "6461", "6464", "9430", "1857".
        label_choice: The organelle. One of "mito" or "lipid_droplet".
        download: Whether to download the data if it is not present.

    Returns:
        The filepaths to the zarr stores, three per sample.
    """
    samples = [samples] if isinstance(samples, str) else samples
    return [p for sample in samples for p in get_parlakgul_liver_eval_data(path, sample, label_choice, download)]


def get_parlakgul_liver_eval_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    samples: Union[str, List[str]] = "6461",
    label_choice: Literal["mito", "lipid_droplet"] = "mito",
    semantic: bool = False,
    return_prediction: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the Parlakgul liver evaluation dataset for mitochondria or lipid droplet segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape (z, y, x). The crops have the shape (25, 150, 150).
        samples: The liver sample(s) to use. One or several of "6461", "6464", "9430", "1857".
        label_choice: The organelle. One of "mito" or "lipid_droplet".
        semantic: Whether to return the binary foreground mask instead of the instance ids.
        return_prediction: Whether to return the model prediction instead of the manual annotation.
            This is meant to compute the evaluation scores of the original segmentations.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    paths = get_parlakgul_liver_eval_paths(path, samples, label_choice, download)

    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)
    if semantic:
        kwargs = util.update_kwargs(kwargs, "label_transform", _binarize)

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key="prediction" if return_prediction else "labels",
        patch_shape=patch_shape,
        **kwargs,
    )


def _binarize(labels):
    return (labels > 0).astype("float32")


def get_parlakgul_liver_eval_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    samples: Union[str, List[str]] = "6461",
    label_choice: Literal["mito", "lipid_droplet"] = "mito",
    semantic: bool = False,
    return_prediction: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for the Parlakgul liver evaluation dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape (z, y, x). The crops have the shape (25, 150, 150).
        batch_size: The batch size.
        samples: The liver sample(s) to use. One or several of "6461", "6464", "9430", "1857".
        label_choice: The organelle. One of "mito" or "lipid_droplet".
        semantic: Whether to return the binary foreground mask instead of the instance ids.
        return_prediction: Whether to return the model prediction instead of the manual annotation.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`
            or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_parlakgul_liver_eval_dataset(
        path, patch_shape, samples=samples, label_choice=label_choice, semantic=semantic,
        return_prediction=return_prediction, download=download, **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
