"""The Wing Disc Timelapse dataset contains annotations for cell instance segmentation in E-cadherin-GFP
spinning disc confocal timelapse movies of cultured Drosophila wing disc epithelia.

The data consists of five movies (biological replicates) of the pouch region of ex vivo cultured wing discs, with
one 2D frame of the apical surface every 5 minutes: 'Ecd_20141010_P2', 'Ecd_20150310_P2', 'Ecd_20150418_P1',
'Ecd_20150608_P1' and 'Ecd_20161213_F0' (see `MOVIES`). Every frame is segmented with TissueAnalyzer and the
segmentation was corrected manually, and the cells were tracked over the movies. Each frame has about 1,800 to 3,000
cells. The movies differ in image size, and 808 frames with segmentation are available in total. Adjacent frames of a
movie are highly correlated, so training and validation data should be split by movie (see the `movies` argument).

The loader exposes the raw E-cadherin frame together with a cell instance label image, where the pixels of the cell
bonds (cell boundaries) are background. The instance ids are consecutive per frame. The label images are derived
from the released 'tracked_cells_resized' images, in which every cell has a unique color, and the 'handCorrection'
bond images. The cell ids are consistent over time in the release, but this module does not keep them, and it does
not use the cell divisions and the tracking database either.

This dataset is different from `torch_em.data.datasets.light_microscopy.wing_disc` (3D nucleus segmentation in
confocal images of wing discs) and `torch_em.data.datasets.light_microscopy.flywing` (128 x 128 patches from an
older epithelial benchmark).

NOTE: The data is stored in one zip archive per movie (2.9 GB in total), and this module converts the frames once
into small tif files. The archives are kept next to the converted data.

The data is located at https://doi.org/10.5281/zenodo.22760592 and is released under a CC-BY-4.0 license.
This dataset is from the publications https://doi.org/10.7554/eLife.57964 and https://doi.org/10.1242/dev.155069.
Please cite them if you use this dataset for your research.
"""

import os
import re
import uuid
from glob import glob
from natsort import natsorted
from concurrent import futures
from typing import Union, Tuple, Optional, Sequence, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL_BASE = "https://zenodo.org/records/22760592/files"

MOVIES = {
    "Ecd_20141010_P2": "0eb41b1e0fbfb23b1af3950c14ad06834c40c6debf639ab572c2ecff006fb478",
    "Ecd_20150310_P2": "07b74f7594a99bd3243e7fde4f1c23967f7256db7db91cccc4a4eb0dd0de4a0c",
    "Ecd_20150418_P1": "dcfc9f79b387b0657723731bcf79fab0d600600161f57577cb54748239a1ef69",
    "Ecd_20150608_P1": "79c52b04b414fc0f4dcb1a73053c2e96f5db02b39b9dff3e0eb25a54e2d25f2d",
    "Ecd_20161213_F0": "8a3ee532c89950bc0f4880ff243593e0c7917f080e352bd41ac6f215ac9da6c0",
}

COMPLETE_MARKER = "complete"
FRAME_PATTERN = re.compile(r".*/Segmentation/[^/]+_(\d+)/(original|tracked_cells_resized|handCorrection)\.(png|tif)$")


def _write_atomic(path, array):
    import tifffile

    tmp_path = f"{path}.{uuid.uuid4().hex}.incomplete.tif"
    tifffile.imwrite(tmp_path, array, compression="zlib")
    os.replace(tmp_path, path)


def _read_member(archive, name):
    from io import BytesIO
    import imageio.v3 as imageio

    return imageio.imread(BytesIO(archive.read(name)), extension=os.path.splitext(name)[1])


def _convert_frame(archive, members, image_path, label_path):
    import numpy as np

    raw = _read_member(archive, members["original"])
    raw = raw[..., 0] if raw.ndim == 3 else raw

    bonds = _read_member(archive, members["handCorrection"])
    bonds = bonds.max(axis=-1) > 0 if bonds.ndim == 3 else bonds > 0

    colors = _read_member(archive, members["tracked_cells_resized"]).astype("int64")
    codes = colors[..., 0] * 65536 + colors[..., 1] * 256 + colors[..., 2]
    codes[bonds] = 0

    foreground = codes > 0
    _, inverse = np.unique(codes[foreground], return_inverse=True)
    labels = np.zeros(codes.shape, dtype="uint16")
    labels[foreground] = inverse + 1
    assert labels.max() < 65535 and raw.shape == labels.shape, f"Unexpected frame '{members['original']}'."

    _write_atomic(image_path, raw.astype("uint8"))
    _write_atomic(label_path, labels)


def _convert_movie(zip_path, out_dir):
    import zipfile

    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "labels"), exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        frames = {}
        for name in archive.namelist():
            match = FRAME_PATTERN.match(name)
            if match is not None:
                frames.setdefault(int(match.group(1)), {})[match.group(2)] = name

        for frame, members in sorted(frames.items()):
            if len(members) < 3:
                continue
            _convert_frame(
                archive, members, os.path.join(out_dir, "images", f"{frame:03d}.tif"),
                os.path.join(out_dir, "labels", f"{frame:03d}.tif"),
            )

    with open(os.path.join(out_dir, COMPLETE_MARKER), "w"):
        pass


def _validate(movies):
    if movies is None:
        return list(MOVIES)
    invalid = [m for m in movies if m not in MOVIES]
    if invalid:
        raise ValueError(f"{invalid} are not valid movies. Choose from {list(MOVIES)}.")
    return list(movies)


def get_wing_disc_timelapse_data(
    path: Union[os.PathLike, str], movies: Optional[Sequence[str]] = None, download: bool = False
) -> str:
    """Download the Wing Disc Timelapse dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        movies: The movies to use. By default all five movies are used. See `MOVIES` for the valid names.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder with the converted frames.
    """
    movies = _validate(movies)
    converted_dir = os.path.join(path, "converted")

    pending = [m for m in movies if not os.path.exists(os.path.join(converted_dir, m, COMPLETE_MARKER))]
    if not pending:
        return converted_dir

    os.makedirs(path, exist_ok=True)
    for movie in pending:
        util.download_source(
            path=os.path.join(path, f"{movie}.zip"), url=f"{URL_BASE}/{movie}.zip", download=download,
            checksum=MOVIES[movie],
        )

    with futures.ProcessPoolExecutor(min(len(pending), os.cpu_count() or 1)) as pool:
        tasks = [
            pool.submit(_convert_movie, os.path.join(path, f"{movie}.zip"), os.path.join(converted_dir, movie))
            for movie in pending
        ]
        for task in tasks:
            task.result()

    return converted_dir


def get_wing_disc_timelapse_paths(
    path: Union[os.PathLike, str], movies: Optional[Sequence[str]] = None, download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the Wing Disc Timelapse data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        movies: The movies to use. By default all five movies are used. See `MOVIES` for the valid names.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    movies = _validate(movies)
    converted_dir = get_wing_disc_timelapse_data(path, movies, download)

    raw_paths, label_paths = [], []
    for movie in movies:
        movie_raw_paths = natsorted(glob(os.path.join(converted_dir, movie, "images", "*.tif")))
        raw_paths.extend(movie_raw_paths)
        label_paths.extend(
            os.path.join(converted_dir, movie, "labels", os.path.basename(p)) for p in movie_raw_paths
        )

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    assert all(os.path.exists(p) for p in label_paths)

    return raw_paths, label_paths


def get_wing_disc_timelapse_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    movies: Optional[Sequence[str]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Wing Disc Timelapse dataset for cell instance segmentation in epithelial timelapse microscopy.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        movies: The movies to use. By default all five movies are used. See `MOVIES` for the valid names.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_wing_disc_timelapse_paths(path, movies, download)

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
        is_seg_dataset=False,
        patch_shape=patch_shape,
        **kwargs
    )


def get_wing_disc_timelapse_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    movies: Optional[Sequence[str]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Wing Disc Timelapse dataloader for cell instance segmentation in epithelial timelapse microscopy.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        movies: The movies to use. By default all five movies are used. See `MOVIES` for the valid names.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_wing_disc_timelapse_dataset(path, patch_shape, movies, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
