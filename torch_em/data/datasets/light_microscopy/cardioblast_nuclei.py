"""The cardioblast nuclei dataset contains annotated time-lapse fluorescence microscopy images.

It shows cardioblast nuclei migrating during Drosophila embryonic development to form the early heart tube.
Each TIFF stack contains a time series of maximum-intensity projections with tracked nucleus instance labels.
One raw movie has 26 unannotated trailing frames; the loader restricts this movie to its annotated prefix.
This loader exposes the subset and train/test split used by https://github.com/kreshuklab/model_ranking.

The dataset is located at https://doi.org/10.6019/S-BIAD1410 and is available under the CC0 license.
It is from the publication https://doi.org/10.1083/jcb.202506102.
Please cite the dataset and publication if you use this dataset in your research.
"""

import os
from typing import List, Literal, Optional, Tuple, Union

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


BASE_URL = "https://www.ebi.ac.uk/biostudies/files/S-BIAD1410/cardioblast_nuclei"

SAMPLES = {
    "train": (
        "cardioblast_nuclei_20200127_e1",
        "cardioblast_nuclei_20200131_e1",
        "cardioblast_nuclei_20200206_e1",
        "cardioblast_nuclei_20200206_e3",
        "cardioblast_nuclei_20220811_e1",
        "cardioblast_nuclei_20220812_e1",
        "cardioblast_nuclei_20220826_e2",
        "cardioblast_nuclei_20220828_e1",
        "cardioblast_nuclei_20220828_e2",
        "cardioblast_nuclei_20220828_e4",
    ),
    "test": (
        "cardioblast_nuclei_20200121_e3",
        "cardioblast_nuclei_20200219_e1",
        "cardioblast_nuclei_20220811_e2",
        "cardioblast_nuclei_20220825_e1",
        "cardioblast_nuclei_20220828_e3",
    ),
}

RAW_CHECKSUMS = {
    "cardioblast_nuclei_20200127_e1": "eaedb22edcef7bfa5b8bb099ed13270b5c7f099178be6c3857341874c5582f8c",
    "cardioblast_nuclei_20200131_e1": "6efa5753a639ea02a627ba2d6fdbd36c13fc42b531881452305e2d769231a40b",
    "cardioblast_nuclei_20200206_e1": "0f347dc898e89ba1f35abd21a7473759a55891db9f79e9062a521cf865677b1f",
    "cardioblast_nuclei_20200206_e3": "0334aff3adec20a8df53b5f628a2523b50d9eee5e8e83797eeb429be4b22d9b3",
    "cardioblast_nuclei_20220811_e1": "6efec42327f5cc81c83534b5762591a36841ca1f4c806ffc2d43abd7eb0b6f38",
    "cardioblast_nuclei_20220812_e1": "452f228ac7e0889eb2ffa7c9a264d372af12f25bf51526380e1ff14300e6367a",
    "cardioblast_nuclei_20220826_e2": "9d7d5e9572b147b47d4a502cc9d06defad67ceda7574b1283d9b77c7e52fbc99",
    "cardioblast_nuclei_20220828_e1": "24d7e20ea7fd268064536f8bc2e3a28cc79a04276fa848e6e029940ef5fb1489",
    "cardioblast_nuclei_20220828_e2": "55a46cbc76e1831c30cfde0e6454168e14c126246ec0cf00e96459e3925a67b3",
    "cardioblast_nuclei_20220828_e4": "82c6cfde80efb1b8416fb8533dd94c73cea227274e8bb7fb3a610309ae7c0620",
    "cardioblast_nuclei_20200121_e3": "dcfd039fe12050c6c75b74fe315fc6cf1f7295fac24f32111e416bb0977f59e2",
    "cardioblast_nuclei_20200219_e1": "b4605fda7b11e091af1e431df81bb96b79e86cb63081e259822d429475f3e869",
    "cardioblast_nuclei_20220811_e2": "a79ff0dcf65d5cf823d1a3f29ed7751f44e08a3545010054b5d95648b8174bd8",
    "cardioblast_nuclei_20220825_e1": "1608f2441090c2f5366e0afbdc980e1c2576444dfbb10bc7b1c9d5ba00fedbb0",
    "cardioblast_nuclei_20220828_e3": "5e8f05ff702a4552495989a20df066322074f571c97797f525e7d8a85e2f5862",
}

LABEL_CHECKSUMS = {
    "cardioblast_nuclei_20200127_e1": "2256bc19a75f7935732dd0ffab43602f5ba58167d828bc0d16c57325929c19a3",
    "cardioblast_nuclei_20200131_e1": "033db5065e39abbadd8d6accd83b330271fd8432fc2cbadeaf680a3bdf8a5f67",
    "cardioblast_nuclei_20200206_e1": "2a7bed91aa3e38542731bd5854abcc2db593dfd1a0329781be52dac39ffd8ece",
    "cardioblast_nuclei_20200206_e3": "6e133822ef6d1ce99b9f7ee66fac16edcb50ef01bd85971cbde3dd2fa6630cee",
    "cardioblast_nuclei_20220811_e1": "1d95565d1d5a925a46e0e8b58055772ed22b8821e6bf5c9d3152e0eeec1764ed",
    "cardioblast_nuclei_20220812_e1": "2224859e84a811210607759ef3c5c46efca07a6ab1f8948889f7e5c4cb0d4a6a",
    "cardioblast_nuclei_20220826_e2": "6dc737a7e0aab77fab1b0489e41fa8eea50ea95cb32866f02c2ad72ae7a78314",
    "cardioblast_nuclei_20220828_e1": "b0f64623fd63ee1f9fdad05fef1bc171d7fed70a470cab68feb236fc0fbff636",
    "cardioblast_nuclei_20220828_e2": "0182987832eba600b9bc2c31048de42d427575abdda8b35809062278f18bb27d",
    "cardioblast_nuclei_20220828_e4": "0f0410ec34e3536beadc7403c98ab7786f87659e31f0739505d807a905ea3292",
    "cardioblast_nuclei_20200121_e3": "a438457b17036de1677557d6c2d56454e36a5e950ca7d3c8e5e263f449962cd0",
    "cardioblast_nuclei_20200219_e1": "a783b5f3c5b460a009f632875f8c1e456903acaf4c2f2531695e8bbd736f808e",
    "cardioblast_nuclei_20220811_e2": "5b6dcdfbc90a0955c129706c3e8defb8003ff6bc17139b03a74a2f81945742c0",
    "cardioblast_nuclei_20220825_e1": "ffcbe838b2887bc8ac02ce0380b768febfcc65e5709abf7b6522481197e5f67f",
    "cardioblast_nuclei_20220828_e3": "bee487d987577f93c7061d884d0fc832bba90854f348b0734208ec631f30b06d",
}

# This movie contains 147 raw frames, but only the first 121 frames are annotated.
ANNOTATED_FRAMES = {"cardioblast_nuclei_20220812_e1": 121}


def get_cardioblast_nuclei_data(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"] = "train",
    download: bool = False,
) -> str:
    """Download the cardioblast nuclei dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the selected data split.
    """
    if split not in SAMPLES:
        raise ValueError(f"'{split}' is not a valid split. Choose from {list(SAMPLES)}.")

    split_dir = os.path.join(path, "cardioblast_nuclei", f"cardioblast_nuclei_{split}")
    for sample in SAMPLES[split]:
        sample_dir = os.path.join(split_dir, sample)
        os.makedirs(sample_dir, exist_ok=True)

        raw_path = os.path.join(sample_dir, f"{sample}.tif")
        label_path = os.path.join(sample_dir, f"{sample}_mask.tif")
        sample_url = f"{BASE_URL}/cardioblast_nuclei_{split}/{sample}"

        util.download_source(raw_path, f"{sample_url}/{sample}.tif", download, checksum=RAW_CHECKSUMS[sample])
        util.download_source(
            label_path, f"{sample_url}/{sample}_mask.tif", download, checksum=LABEL_CHECKSUMS[sample]
        )

    return split_dir


def get_cardioblast_nuclei_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"] = "train",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the cardioblast images and nucleus instance labels.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        The image paths and corresponding label paths.
    """
    split_dir = get_cardioblast_nuclei_data(path, split, download)
    raw_paths = [os.path.join(split_dir, sample, f"{sample}.tif") for sample in SAMPLES[split]]
    label_paths = [os.path.join(split_dir, sample, f"{sample}_mask.tif") for sample in SAMPLES[split]]

    missing_paths = [path for path in raw_paths + label_paths if not os.path.exists(path)]
    if missing_paths:
        raise RuntimeError(f"Could not find {len(missing_paths)} cardioblast nuclei files for split '{split}'.")

    return raw_paths, label_paths


def get_cardioblast_nuclei_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"] = "train",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the cardioblast nuclei dataset for instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The 2D patch shape to use for training.
        split: The data split. Either 'train' or 'test'.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if len(patch_shape) != 2:
        raise ValueError(f"The cardioblast nuclei patch shape must be two-dimensional, got {patch_shape}.")

    raw_paths, label_paths = get_cardioblast_nuclei_paths(path, split, download)
    rois = [
        (slice(0, ANNOTATED_FRAMES.get(sample)), slice(None), slice(None))
        for sample in SAMPLES[split]
    ]
    kwargs.setdefault("rois", rois)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, offsets=offsets, boundaries=boundaries, binary=binary,
    )
    kwargs = util.ensure_transforms(ndim=2, **kwargs)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=(1,) + patch_shape,
        is_seg_dataset=True,
        ndim=2,
        **kwargs,
    )


def get_cardioblast_nuclei_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"] = "train",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the cardioblast nuclei dataloader for instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The 2D patch shape to use for training.
        split: The data split. Either 'train' or 'test'.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cardioblast_nuclei_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
