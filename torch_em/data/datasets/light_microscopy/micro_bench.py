"""Micro-Bench is a vision-language benchmark assembled from microscopy datasets.

This loader exposes the polygon segmentation annotations redistributed with Micro-Bench for
synthetic cells from Burgess et al., CellCognition H2B images, OpenCell nuclei and mitochondria
from Wu et al. The GlaS subset is not included because torch-em already provides the original
data via ``get_glas_loader``.

The Micro-Bench release is available under CC BY-SA 4.0 at
https://huggingface.co/datasets/jnirschl/uBench. The source metadata reports CC BY 4.0 for the
Burgess and OpenCell samples and CC BY-SA 4.0 for the Wu samples, but does not specify a license
for CellCognition. The benchmark is from the publication https://doi.org/10.52202/079017-0965.
Please cite it and the corresponding source dataset if you use these data in your research.
"""

import os
from glob import glob
from io import BytesIO
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import numpy as np

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


REVISION = "3f2c5b590bc7a208d5b60f3527ce4c76a331aa2b"
URL = (
    "https://huggingface.co/datasets/jnirschl/uBench/resolve/"
    f"{REVISION}/perception/0.1.0/{{filename}}?download=true"
)

FILES = {
    "ubench-test-00000-of-00007.arrow": "3d346134e6dffed41f74da64c1c173f5efd3a10200805796a030357297b630f1",
    "ubench-test-00001-of-00007.arrow": "790591636ae679afa1161664d207c2bb2e3d583690d03967e38364ba0f31ccac",
    "ubench-test-00002-of-00007.arrow": "c370cd84e6ed9b00483ef3d20f5c06d5ff2264733ea87f2df98cf9a22820bf74",
    "ubench-test-00003-of-00007.arrow": "832cdd6f73a132028584b4e47ec1afe643e0490dd8ec9686eabbb9de3eb7f814",
    "ubench-test-00004-of-00007.arrow": "e2741415cef1327ba243f21000ac43aab2d36b61a9c2c862a1d0e21f200f4a69",
    "ubench-test-00005-of-00007.arrow": "89deae84ed2dab1bb1fc40be0aefe81b8d94590cec548d1950ff5ff853143d34",
    "ubench-test-00006-of-00007.arrow": "534beca2084a250d4436340c78accc9fe7cc2d6063d4884302e7e7f141edad6c",
}

SOURCES = ("burgess", "cellcognition", "opencell", "wu")


def _get_source(row) -> Optional[str]:
    dataset = row["dataset"]
    if dataset is not None and dataset.startswith("burgess_et_al_2024_"):
        return "burgess"
    if dataset == "opencell":
        return "opencell"
    if dataset == "wu_et_al_2023":
        return "wu"

    classes = {annotation["className"] for annotation in row["polygon"] or []}
    if dataset is None and "H2B-mCherry" in classes:
        return "cellcognition"
    return None


def _get_sample_id(row, source: str) -> str:
    dataset = source if row["dataset"] is None else row["dataset"]
    return f"{dataset}_{row['image_id']}"


def _rasterize(polygons, shape: Tuple[int, int], label_choice: Optional[str] = None) -> np.ndarray:
    from skimage.draw import polygon as draw_polygon

    labels = np.zeros(shape, dtype="uint16")
    instance_id = 0
    for annotation in polygons:
        if label_choice is not None and annotation["className"] != label_choice:
            continue

        points = np.asarray(annotation["points"], dtype=float).reshape(-1, 2)
        rows, columns = draw_polygon(points[:, 1], points[:, 0], shape=shape)
        instance_id += 1
        labels[rows, columns] = instance_id
    return labels


def _process_shard(path: Union[os.PathLike, str], arrow_path: Union[os.PathLike, str]) -> None:
    import imageio.v3 as imageio
    import pyarrow.ipc as ipc
    from PIL import Image
    from tqdm import tqdm

    with open(arrow_path, "rb") as file:
        reader = ipc.open_stream(file)
        for batch in tqdm(reader, desc=f"Process {Path(arrow_path).name}"):
            columns = batch.select(["image_id", "image", "dataset", "polygon"])
            for row in columns.to_pylist():
                source = _get_source(row)
                if source is None:
                    continue

                image = np.asarray(Image.open(BytesIO(row["image"]["bytes"])).convert("RGB"))
                sample_id = _get_sample_id(row, source)
                image_path = os.path.join(path, "images", source, f"{sample_id}.tif")
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                if not os.path.exists(image_path):
                    imageio.imwrite(image_path, image, compression="zlib")

                choices = ("cell", "nucleus") if source == "burgess" else ("instances",)
                for choice in choices:
                    label_path = os.path.join(path, "labels", source, choice, f"{sample_id}.tif")
                    if os.path.exists(label_path):
                        continue

                    os.makedirs(os.path.dirname(label_path), exist_ok=True)
                    label_choice = choice if source == "burgess" else None
                    labels = _rasterize(row["polygon"], image.shape[:2], label_choice)
                    if labels.max() == 0:
                        raise RuntimeError(f"No '{choice}' polygons found for Micro-Bench sample {sample_id}.")
                    imageio.imwrite(label_path, labels, compression="zlib")


def get_micro_bench_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download and prepare the segmentation subset of Micro-Bench.

    The seven official test shards contain 3.57 GB in total. Each shard is removed after its
    segmentation samples have been extracted, so at most one source shard is stored alongside
    the prepared data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder with the prepared data.
    """
    marker_dir = os.path.join(path, ".processed")
    os.makedirs(marker_dir, exist_ok=True)

    for filename, checksum in FILES.items():
        marker_path = os.path.join(marker_dir, filename)
        if os.path.exists(marker_path):
            continue

        arrow_path = os.path.join(path, filename)
        util.download_source(arrow_path, URL.format(filename=filename), download, checksum)
        _process_shard(path, arrow_path)
        Path(marker_path).touch()
        os.remove(arrow_path)

    return str(path)


def _validate_source(source: str, label_choice: Optional[str]) -> str:
    if source not in SOURCES:
        raise ValueError(f"'{source}' is not a valid source. Choose from {list(SOURCES)}.")

    if source == "burgess":
        label_choice = "cell" if label_choice is None else label_choice
        if label_choice not in ("cell", "nucleus"):
            raise ValueError("Burgess annotations support label_choice 'cell' or 'nucleus'.")
    elif label_choice is not None:
        raise ValueError("label_choice is only supported for source='burgess'.")
    else:
        label_choice = "instances"
    return label_choice


def get_micro_bench_paths(
    path: Union[os.PathLike, str],
    source: Literal["burgess", "cellcognition", "opencell", "wu"] = "burgess",
    label_choice: Optional[Literal["cell", "nucleus"]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to one segmentation source in Micro-Bench.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        source: The source dataset. Choose from 'burgess', 'cellcognition', 'opencell' or 'wu'.
        label_choice: The Burgess target, either 'cell' or 'nucleus'. The two targets are kept
            separate because their polygons overlap. This argument is only valid for Burgess.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the instance label data.
    """
    from natsort import natsorted

    label_choice = _validate_source(source, label_choice)
    get_micro_bench_data(path, download)

    image_paths = natsorted(glob(os.path.join(path, "images", source, "*.tif")))
    label_paths = natsorted(glob(os.path.join(path, "labels", source, label_choice, "*.tif")))
    if not image_paths or len(image_paths) != len(label_paths):
        raise RuntimeError(f"Could not find matching Micro-Bench images and labels for source '{source}' in {path}.")
    return image_paths, label_paths


def get_micro_bench_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    source: Literal["burgess", "cellcognition", "opencell", "wu"] = "burgess",
    label_choice: Optional[Literal["cell", "nucleus"]] = None,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get a Micro-Bench instance segmentation dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        source: The source dataset. Choose from 'burgess', 'cellcognition', 'opencell' or 'wu'.
        label_choice: The Burgess target, either 'cell' or 'nucleus'. This is only valid for Burgess.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for ``torch_em.default_segmentation_dataset``.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_micro_bench_paths(path, source, label_choice, download)
    kwargs, _ = util.add_instance_label_transform(kwargs, add_binary_target=True)
    kwargs = util.update_kwargs(kwargs, "ndim", 2)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs,
    )


def get_micro_bench_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    source: Literal["burgess", "cellcognition", "opencell", "wu"] = "burgess",
    label_choice: Optional[Literal["cell", "nucleus"]] = None,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get a Micro-Bench instance segmentation data loader.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        source: The source dataset. Choose from 'burgess', 'cellcognition', 'opencell' or 'wu'.
        label_choice: The Burgess target, either 'cell' or 'nucleus'. This is only valid for Burgess.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for the dataset or PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_micro_bench_dataset(
        path, patch_shape, source, label_choice, download, **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
