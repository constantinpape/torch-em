"""The BBBC046 dataset (FiloData3D) contains synthetic 3D time-lapse fluorescence microscopy images of
single A549 lung cancer cells with filopodia, and ground truth masks of the cell body and filopodia.

The dataset consists of 180 synthetic sequences of 30 frames each (5400 volumes). They are derived from
9 base sequences, 3 per cell phenotype: wild-type ('WT-ID550', 'WT-ID551', 'WT-ID552'),
CRMP-2-overexpressing ('OE-ID350', 'OE-ID351', 'OE-ID352') and CRMP-2-phospho-defective
('PD-ID450', 'PD-ID451', 'PD-ID452'). Every base sequence is rendered for 4 anisotropy ratios
(1, 2, 4 and 8, which reduce the number of z-slices) and 5 fluorescence level factors
(0.25, 0.50, 1.00, 2.00 and 4.00, which change the signal-to-noise ratio).
The ground truth masks are shared across the fluorescence level factors of a sequence.

The label ids are: 0 (background), 50 (cell body) and ids of 100 and above for the filopodia. The k-th filopodium
uses the ids 100 * k (its primary branch) and 100 * k + 1, 100 * k + 2, ... (its side branches). These ids are also
used in the accompanying trajectory and length text files.

The base sequences are downloaded separately, each one as an archive of 1.2 to 9.8 GB (36 GB in total).

The dataset is located at https://bbbc.broadinstitute.org/BBBC046.

This dataset is from the publications https://doi.org/10.1109/TMI.2018.2845884 and
https://doi.org/10.1109/ICIP.2019.8803721.
Please cite them if you use this dataset in your research.
"""

import os
import shutil
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List, Optional, Sequence

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


SEQUENCE_IDS = [
    "OE-ID350", "OE-ID351", "OE-ID352", "PD-ID450", "PD-ID451", "PD-ID452", "WT-ID550", "WT-ID551", "WT-ID552",
]

URLS = {sequence_id: f"https://data.broadinstitute.org/bbbc/BBBC046/{sequence_id}.zip" for sequence_id in SEQUENCE_IDS}

CHECKSUMS = {
    "OE-ID350": "ee22ecdb33311be4533b00bae4866604e1062466056a46a575486bdfefe2c487",
    "OE-ID351": "da496e9018c584ea15a7b12fca6e625b448f5acb9b844782755dadd443000a95",
    "OE-ID352": "cb0ae686a980abb2318e03513420ed007f0b30044079778832809b64dd98f4eb",
    "PD-ID450": "b67cc148bf38674e3cbecb4f7b044dec3a1b32a3363c53d14baeab7fbb88a0dc",
    "PD-ID451": "ce39ad00131beda3f41e886377ac2030d866213ff8b39f302b15c493da5086c0",
    "PD-ID452": "3f0fb99a96af7299f9029a93ec867be0e94cd18daae8919a40cfa9cfaa82a924",
    "WT-ID550": "2dcf6dbd1faff96fb919aeb7249cf1f25196729ff8304f556d5275a223af878b",
    "WT-ID551": "c6e3216bc50ce76d66bd4a35e2d452528aa939852f6f513cf3586dc8e2603e4f",
    "WT-ID552": "222c35d171f9c3f1808f302bc9f53700af99267bfede5f54701918d929559f7a",
}

ANISOTROPY_RATIOS = [1, 2, 4, 8]
FLUORESCENCE_FACTORS = ["0.25", "0.50", "1.00", "2.00", "4.00"]


def _unzip_with_offset_fix(zip_path, dst):
    """Extract a zip archive whose central directory stores wrong local header offsets.

    The archives of the 'PD' sequences (larger than 4 GB) were created with a tool that wrote local header offsets
    shifted by a multiple of 4 GB, which `zipfile` cannot read directly. We correct the offsets before extraction.
    """
    import zipfile

    with zipfile.ZipFile(zip_path) as f:
        infos = f.infolist()
        for info in infos:
            for shift in (0, -2**32, 2**32, -2**33, 2**33):
                offset = info.header_offset + shift
                if offset < 0:
                    continue
                f.fp.seek(offset)
                if f.fp.read(4) == b"PK\x03\x04":
                    info.header_offset = offset
                    break
            else:
                raise RuntimeError(f"Could not locate the local header of '{info.filename}' in '{zip_path}'.")

        # Recent python versions check for overlapping entries with the end offsets computed when opening
        # the archive. We recompute them for the corrected header offsets.
        infos = sorted(infos, key=lambda info: info.header_offset)
        for info, next_info in zip(infos, infos[1:] + [None]):
            if hasattr(info, "_end_offset"):
                info._end_offset = f.start_dir if next_info is None else next_info.header_offset

        for info in infos:
            f.extract(info, dst)

    os.remove(zip_path)


def get_bbbc046_data(path: Union[os.PathLike, str], sequence_id: str, download: bool = False) -> List[str]:
    """Download one base sequence of the BBBC046 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        sequence_id: The base sequence to download. One of the ids in `SEQUENCE_IDS`.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the sequence folders, one per anisotropy ratio.
    """
    if sequence_id not in SEQUENCE_IDS:
        raise ValueError(f"'{sequence_id}' is not a valid sequence id. Choose one of {SEQUENCE_IDS}.")

    sequence_dirs = natsorted(glob(os.path.join(path, f"{sequence_id}-AR-*")))
    if len(sequence_dirs) == len(ANISOTROPY_RATIOS):
        return sequence_dirs

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, f"{sequence_id}.zip")
    util.download_source(path=zip_path, url=URLS[sequence_id], download=download, checksum=CHECKSUMS[sequence_id])
    _unzip_with_offset_fix(zip_path, path)

    # Some archives (e.g. 'OE-ID350') contain one nested archive per anisotropy ratio, which we extract as well.
    for nested_zip_path in natsorted(glob(os.path.join(path, f"{sequence_id}-AR-*.zip"))):
        util.unzip(zip_path=nested_zip_path, dst=path)

    # The archives contain macOS metadata folders which we do not need.
    macos_dir = os.path.join(path, "__MACOSX")
    if os.path.exists(macos_dir):
        shutil.rmtree(macos_dir)

    sequence_dirs = natsorted(glob(os.path.join(path, f"{sequence_id}-AR-*")))
    assert len(sequence_dirs) == len(ANISOTROPY_RATIOS), f"Unexpected folder structure for '{sequence_id}'."

    return sequence_dirs


def get_bbbc046_paths(
    path: Union[os.PathLike, str],
    sequence_ids: Optional[Sequence[str]] = None,
    anisotropy_ratio: Optional[Literal[1, 2, 4, 8]] = None,
    fluorescence_factor: Optional[Literal["0.25", "0.50", "1.00", "2.00", "4.00"]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the BBBC046 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        sequence_ids: The base sequences to use. By default, all 9 base sequences are used.
        anisotropy_ratio: The anisotropy ratio to use. By default, all 4 anisotropy ratios are used.
        fluorescence_factor: The fluorescence level factor to use. By default, all 5 factors are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if sequence_ids is None:
        sequence_ids = SEQUENCE_IDS
    elif isinstance(sequence_ids, str):
        sequence_ids = [sequence_ids]

    if anisotropy_ratio is not None and anisotropy_ratio not in ANISOTROPY_RATIOS:
        raise ValueError(f"'{anisotropy_ratio}' is not a valid anisotropy ratio. Choose one of {ANISOTROPY_RATIOS}.")
    if fluorescence_factor is not None and fluorescence_factor not in FLUORESCENCE_FACTORS:
        raise ValueError(
            f"'{fluorescence_factor}' is not a valid fluorescence factor. Choose one of {FLUORESCENCE_FACTORS}."
        )

    ar_pattern = "*" if anisotropy_ratio is None else str(anisotropy_ratio)
    factor_pattern = "*" if fluorescence_factor is None else fluorescence_factor

    raw_paths, label_paths = [], []
    for sequence_id in sequence_ids:
        get_bbbc046_data(path, sequence_id, download)
        pattern = os.path.join(path, f"{sequence_id}-AR-{ar_pattern}", f"factor-{factor_pattern}", "img_t*.tif")
        curr_raw_paths = natsorted(glob(pattern))
        assert len(curr_raw_paths) > 0, f"No volumes found for '{sequence_id}' with the pattern '{pattern}'."

        for raw_path in curr_raw_paths:
            label_path = os.path.join(
                os.path.dirname(os.path.dirname(raw_path)), os.path.basename(raw_path).replace("img_", "mask_")
            )
            assert os.path.exists(label_path), f"The label volume '{label_path}' is missing."
            raw_paths.append(raw_path)
            label_paths.append(label_path)

    return raw_paths, label_paths


def get_bbbc046_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    sequence_ids: Optional[Sequence[str]] = None,
    anisotropy_ratio: Optional[Literal[1, 2, 4, 8]] = None,
    fluorescence_factor: Optional[Literal["0.25", "0.50", "1.00", "2.00", "4.00"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the BBBC046 dataset for cell body and filopodia segmentation in synthetic 3D time-lapse images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        sequence_ids: The base sequences to use. By default, all 9 base sequences are used.
        anisotropy_ratio: The anisotropy ratio to use. By default, all 4 anisotropy ratios are used.
        fluorescence_factor: The fluorescence level factor to use. By default, all 5 factors are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_bbbc046_paths(path, sequence_ids, anisotropy_ratio, fluorescence_factor, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_bbbc046_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    sequence_ids: Optional[Sequence[str]] = None,
    anisotropy_ratio: Optional[Literal[1, 2, 4, 8]] = None,
    fluorescence_factor: Optional[Literal["0.25", "0.50", "1.00", "2.00", "4.00"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the BBBC046 dataloader for cell body and filopodia segmentation in synthetic 3D time-lapse images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        sequence_ids: The base sequences to use. By default, all 9 base sequences are used.
        anisotropy_ratio: The anisotropy ratio to use. By default, all 4 anisotropy ratios are used.
        fluorescence_factor: The fluorescence level factor to use. By default, all 5 factors are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_bbbc046_dataset(
        path, patch_shape, sequence_ids, anisotropy_ratio, fluorescence_factor, resize_inputs, download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
