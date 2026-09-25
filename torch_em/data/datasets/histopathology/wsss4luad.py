"""WSSS4LUAD is a histopathology dataset for weakly-supervised tissue semantic segmentation
in H&E-stained lung adenocarcinoma whole-slide images, with tumor epithelial tissue, tumor-
associated stroma, and normal tissue as the tissue classes.

The original challenge data only provides patch-level (image-level) classification labels
for the training split. Pixel-level ground-truth segmentation masks are only provided for
the validation and test splits, which is why this loader restricts itself to these two splits.
Note that 10 of the 80 test patches have no mask in the mirrored data (their ground-truth
was withheld for the original challenge leaderboard); these are skipped automatically.

The data is mirrored on Hugging Face at https://huggingface.co/datasets/Angelou0516/WSSS4LUAD-v2
(the original challenge data at https://wsss4luad.grand-challenge.org/WSSS4LUAD/ requires
requesting access via email, as the challenge has closed). The dataset is licensed under
CC BY 4.0. This dataset is from the publication https://doi.org/10.48550/arXiv.2204.06455.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from typing import List, Literal, Tuple, Union

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


HF_REPO = "Angelou0516/WSSS4LUAD-v2"

SPLITS = {"val": "validation.parquet", "test": "test.parquet"}


def _extract_split(path, split):
    import io
    import numpy as np
    import pyarrow.parquet as pq
    import imageio.v3 as imageio
    from PIL import Image
    from tqdm import tqdm

    image_dir = os.path.join(path, "images", split)
    mask_dir = os.path.join(path, "masks", split)
    if os.path.exists(image_dir) and os.path.exists(mask_dir):
        image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
        if len(image_paths) > 0:
            return

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    parquet_path = os.path.join(path, SPLITS[split])
    table = pq.read_table(parquet_path)

    for row in tqdm(table.to_pylist(), desc=f"Extracting WSSS4LUAD '{split}' split"):
        # A subset of the 'test' split rows have no mask in the mirrored parquet files
        # (the corresponding ground-truth was withheld for the original challenge leaderboard).
        if row["mask"] is None:
            continue

        name = os.path.splitext(row["filename"])[0]

        image_out = os.path.join(image_dir, f"{name}.png")
        if not os.path.exists(image_out):
            image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
            image.save(image_out)

        mask_out = os.path.join(mask_dir, f"{name}.tif")
        if not os.path.exists(mask_out):
            mask = np.array(Image.open(io.BytesIO(row["mask"]["bytes"])))
            imageio.imwrite(mask_out, mask.astype("uint8"), compression="zlib")


def get_wsss4luad_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the WSSS4LUAD validation and test splits (the only splits with pixel-level masks).

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder where the data is stored.
    """
    os.makedirs(path, exist_ok=True)

    missing_splits = [
        split for split in SPLITS
        if not (os.path.exists(os.path.join(path, "images", split)) and glob(
            os.path.join(path, "images", split, "*.png")
        ))
    ]
    if not missing_splits:
        return path

    if not download:
        raise RuntimeError(f"Cannot find the data at {path}, but 'download' is set to False.")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("'huggingface_hub' is required to download this dataset.")

    for split in missing_splits:
        hf_hub_download(
            repo_id=HF_REPO, filename=SPLITS[split], repo_type="dataset", local_dir=path,
        )
        _extract_split(path, split)

    return path


def get_wsss4luad_paths(
    path: Union[os.PathLike, str], split: Literal["val", "test"], download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the WSSS4LUAD image and tissue segmentation mask data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. Either 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    assert split in SPLITS, f"'{split}' is not a valid split. Choose from {list(SPLITS.keys())}."
    data_dir = get_wsss4luad_data(path, download)

    image_paths = sorted(glob(os.path.join(data_dir, "images", split, "*.png")))
    label_paths = sorted(glob(os.path.join(data_dir, "masks", split, "*.tif")))
    assert len(image_paths) == len(label_paths) and len(image_paths) > 0

    return image_paths, label_paths


def get_wsss4luad_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the WSSS4LUAD dataset for tissue semantic segmentation.

    The masks use label 0 for tumor epithelial tissue, 1 for tumor-associated stroma,
    2 for normal tissue, and 3 for background / excluded pixels (e.g. white alveolar
    space), which should typically not be used for computing losses or metrics.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'val' or 'test'.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_wsss4luad_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        ndim=2,
        with_channels=True,
        **kwargs,
    )


def get_wsss4luad_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the WSSS4LUAD dataloader for tissue semantic segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'val' or 'test'.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_wsss4luad_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
