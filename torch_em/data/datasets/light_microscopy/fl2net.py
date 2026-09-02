"""The FL2-Net dataset contains 3D bright-field microscopy timelapses of mouse embryos
with nuclear instance segmentation annotations.

The dataset holds 84 embryos with 506 timepoints each, so 42504 volumes in total. The authors split
the data by embryo, and no embryo appears in more than one split. The images are label-free, which
makes the nuclei much harder to see than in a fluorescence image.

NOTE: Every volume holds 51 z-slices, but each embryo was cropped to its own field of view, so the
xy shape differs between embryos. It ranges from (92, 102) to (158, 158), and is not square for most
embryos. Volumes smaller than `patch_shape` are padded, so keep the last two entries of your
`patch_shape` at 92 or below to train on unpadded data.

NOTE: The volumes are stored as uncompressed uint16 tif files of about 1.8 MB each, so extracted in
full each of the two archives needs about 76 GB. The loader therefore extracts only the timepoints
that you request. Use `stride` to set how many timepoints it skips, or pass `timepoints` to select
them. The default stride of 25 keeps 21 of the 506 timepoints, which comes out at about 2 GB for the
train split.

NOTE: The archives store their files in an arbitrary order, so extracting even a few timepoints
reads through the whole archive once. This takes about a minute for the annotations and much longer
for the images. The files are cached on disk, so this cost is only paid the first time.

NOTE: The images take up 64 GiB as a single archive, so expect the download to run for a while.
Google Drive refuses that archive to anonymous callers with a 'quota exceeded' html page, which it
serves under HTTP 200 rather than as an error. It does answer ranged requests though, so this module
downloads the archives range by range, which also lets an interrupted download resume. If the
download fails anyway, fetch both archives manually from the links in
https://github.com/funalab/FL2-Net and place them in `path` as 'raw.tar.gz' and 'gt.tar.gz'.

The dataset is located at https://github.com/funalab/FL2-Net.
This dataset is from the publication https://doi.org/10.1016/j.compbiomed.2025.111179.
Please cite it if you use this dataset in your research.
"""

import os
import re
import tarfile
from tqdm import tqdm
from typing import List, Literal, Optional, Sequence, Tuple, Union

import requests

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


# These are the links from the dataset section of the FL2-Net README. Google Drive answers them with
# a 'quota exceeded' html page rather than the file, so the download below addresses the files by id
# instead. The urls are kept for reference and for the manual download instructions.
URLS = {
    "images": "https://drive.usercontent.google.com/download?id=1OAMmFM76TputGnU6nell6LU81N0hDmRc&confirm=xxx",
    "labels": "https://drive.usercontent.google.com/download?id=1hdSnCthLtyKMCahFLHUz36Awtj2-OC6T&confirm=xxx",
}

FILE_IDS = {"images": "1OAMmFM76TputGnU6nell6LU81N0hDmRc", "labels": "1hdSnCthLtyKMCahFLHUz36Awtj2-OC6T"}

CHECKSUMS = {
    "images": None,  # Filled in once the 64 GiB download has completed and been verified.
    "labels": "9c12b70978f3995662f377dac8fc173abdc0a350ee3c38e6367096c87c2d2200",
}

# The size of the archives in bytes, as reported by Google Drive. The download checks against these,
# because a truncated transfer is otherwise only caught by the much slower checksum.
ARCHIVE_SIZES = {"images": 69032222254, "labels": 456938451}

MANUAL_DOWNLOAD_MESSAGE = (
    "Google Drive refused to serve the FL2-Net archives, because too many users have downloaded them "
    "recently. Please download '{name}' manually from the dataset links in "
    "https://github.com/funalab/FL2-Net, save it as '{path}', and run this function again."
)

ARCHIVE_NAMES = {"images": "raw.tar.gz", "labels": "gt.tar.gz"}

N_TIMEPOINTS = 506

# The authors split the data by embryo in datasets/split_list_411 of the FL2-Net repository.
SPLITS = {
    "train": (
        "F001/Embryo01", "F001/Embryo02", "F001/Embryo03", "F001/Embryo04",
        "F001/Embryo06", "F001/Embryo08", "F001/Embryo09", "F001/Embryo10",
        "F002/Embryo01", "F002/Embryo03", "F002/Embryo04", "F002/Embryo06",
        "F002/Embryo07", "F002/Embryo08", "F002/Embryo10", "F002/Embryo11",
        "F003/Embryo01", "F003/Embryo02", "F003/Embryo04", "F003/Embryo05",
        "F003/Embryo08", "F003/Embryo09", "F003/Embryo10", "F003/Embryo12",
        "F004/Embryo01", "F004/Embryo02", "F004/Embryo05", "F004/Embryo06",
        "F004/Embryo08", "F004/Embryo09", "F004/Embryo10", "F004/Embryo12",
        "F005/Embryo02", "F005/Embryo04", "F005/Embryo05", "F005/Embryo06",
        "F005/Embryo08", "F005/Embryo09", "F005/Embryo10", "F005/Embryo11",
        "F006/Embryo01", "F006/Embryo04", "F006/Embryo05", "F006/Embryo08",
        "F006/Embryo09", "F006/Embryo10", "F006/Embryo11", "F006/Embryo12",
        "F007/Embryo03", "F007/Embryo04", "F007/Embryo05", "F007/Embryo06",
        "F007/Embryo07", "F007/Embryo09", "F007/Embryo10", "F007/Embryo11",
    ),
    "val": (
        "F001/Embryo05", "F001/Embryo12", "F002/Embryo05", "F002/Embryo09",
        "F003/Embryo03", "F003/Embryo06", "F004/Embryo04", "F004/Embryo11",
        "F005/Embryo01", "F005/Embryo03", "F006/Embryo06", "F006/Embryo07",
        "F007/Embryo01", "F007/Embryo12",
    ),
    "test": (
        "F001/Embryo07", "F001/Embryo11", "F002/Embryo02", "F002/Embryo12",
        "F003/Embryo07", "F003/Embryo11", "F004/Embryo03", "F004/Embryo07",
        "F005/Embryo07", "F005/Embryo12", "F006/Embryo02", "F006/Embryo03",
        "F007/Embryo02", "F007/Embryo08",
    ),
}


def _get_download_url(session: requests.Session, file_id: str) -> str:
    """Get a download url for a large Google Drive file, and store the matching cookie in `session`.

    Google cannot virus scan files of this size, so it answers with an interstitial page that holds a
    confirmation token. That token, together with the cookie, is what makes the file downloadable.
    """
    response = session.get(f"https://drive.google.com/uc?export=download&id={file_id}", timeout=120)
    response.raise_for_status()
    token = re.search(r'name="uuid" value="([^"]+)"', response.text)
    if token is None:
        raise RuntimeError(
            "Google Drive did not return a download token for the FL2-Net archive. "
            f"It answered with: {response.text[:200]!r}"
        )
    return (
        f"https://drive.usercontent.google.com/download?id={file_id}"
        f"&export=download&confirm=t&uuid={token.group(1)}"
    )


def _download_from_gdrive(path: str, file_id: str, total: int, checksum: Optional[str], desc: str) -> None:
    """Download a large public Google Drive file in chunks, and resume an interrupted download.

    Google refuses these files to anonymous callers with a 'quota exceeded' html page under HTTP 200,
    so a plain download writes that page to disk instead of the file. A ranged request for the same
    url is served normally, so this reads the file range by range. That also makes the download
    resumable, which matters for an archive of this size.

    Ranges of more than 512 MiB are refused the same way as an unranged request, so the chunk size
    stays well below that, and halves whenever a chunk is refused in case the limit is lowered.
    """
    chunk_size = 256 * 1024**2
    min_chunk_size = 32 * 1024**2
    tmp_path = f"{path}.incomplete"
    session = requests.Session()
    url = _get_download_url(session, file_id)

    with tqdm(total=total, unit="B", unit_scale=True, desc=desc) as progress:
        offset = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
        progress.update(offset)

        while offset < total:
            end = min(offset + chunk_size, total) - 1
            expected = end - offset + 1
            response = session.get(url, headers={"Range": f"bytes={offset}-{end}"}, stream=True, timeout=3600)

            # A refusal comes back as HTTP 200 with an html body rather than as an error code, so the
            # status and the length are both checked before anything is written.
            if response.status_code != 206 or int(response.headers.get("Content-Length", 0)) != expected:
                response.close()
                if chunk_size > min_chunk_size:
                    chunk_size //= 2
                url = _get_download_url(session, file_id)
                continue

            with open(tmp_path, "ab") as f:
                for chunk in response.iter_content(chunk_size=1024**2):
                    f.write(chunk)
                    progress.update(len(chunk))
            offset = os.path.getsize(tmp_path)

    if offset != total:
        raise RuntimeError(f"Downloaded {offset} bytes of {path}, but expected {total}.")

    util._check_checksum(tmp_path, checksum)
    os.replace(tmp_path, path)


def _get_archive_root(archive_path: str) -> str:
    """Read the name of the top level folder of an archive.

    The two archives do not agree on this name: the images sit under 'raw' and the annotations under
    'qcanet'. Neither is documented, so it is read here rather than assumed.
    """
    # 'r|gz' reads the archive as a stream. That is all this needs, and it avoids the seeks that
    # 'r:gz' performs for every member, which are expensive on a gzip stream of this size.
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            root = member.name.split("/")[0]
            if root:
                return root
    raise RuntimeError(f"The archive {archive_path} is empty.")


def _extract_members(archive_path: str, relative_names: Sequence[str], destination: str) -> None:
    """Extract the given files from an archive in one pass, and drop the top level folder.

    The archives store their files in an arbitrary order, so the whole archive has to be read to
    find the requested ones. Files that were extracted before are skipped.
    """
    missing = {name for name in relative_names if not os.path.exists(os.path.join(destination, name))}
    if not missing:
        return

    root = _get_archive_root(archive_path)
    wanted = {f"{root}/{name}": name for name in missing}

    found = set()
    desc = f"Extract {len(missing)} files from {os.path.basename(archive_path)}"
    with tarfile.open(archive_path, "r|gz") as archive:
        with tqdm(total=len(wanted), desc=desc) as progress:
            try:
                for member in archive:
                    target = wanted.get(member.name)
                    if target is None:
                        continue
                    output_path = os.path.join(destination, target)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    # Write to a temporary path first, so that an interrupted extraction is not
                    # mistaken for a complete one when this function is called again.
                    tmp_path = f"{output_path}.incomplete"
                    with archive.extractfile(member) as source, open(tmp_path, "wb") as f:
                        f.write(source.read())
                    os.replace(tmp_path, output_path)
                    found.add(target)
                    progress.update(1)
                    if len(found) == len(wanted):
                        break
            except (tarfile.ReadError, EOFError) as e:
                # An incomplete download ends mid-stream. Everything up to that point was extracted,
                # so say what happened rather than letting a bare gzip error surface.
                raise RuntimeError(
                    f"The archive {archive_path} ends before its end-of-stream marker, so the "
                    f"download is incomplete. {len(found)} of {len(missing)} requested files were "
                    f"found before it broke off. Delete the archive to download it again, or fetch "
                    f"it manually from https://github.com/funalab/FL2-Net. The original error was: {e}"
                ) from e

    if found != missing:
        raise RuntimeError(
            f"Could not find {len(missing - found)} of {len(missing)} files in {archive_path}. "
            f"The first missing file is '{sorted(missing - found)[0]}'."
        )


def _get_timepoints(timepoints: Optional[Sequence[int]], stride: int) -> List[int]:
    """Resolve the requested timepoints. The timepoint index starts at one."""
    if timepoints is not None:
        selected = sorted({int(t) for t in timepoints})
        if not selected:
            raise ValueError("You have to request at least one timepoint.")
        for timepoint in selected:
            if not 1 <= timepoint <= N_TIMEPOINTS:
                raise ValueError(f"The timepoint {timepoint} is outside the range 1 to {N_TIMEPOINTS}.")
        return selected

    if stride < 1:
        raise ValueError(f"The stride must be at least one, got {stride}.")
    return list(range(1, N_TIMEPOINTS + 1, stride))


def get_fl2net_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the FL2-Net dataset.

    NOTE: The image archive is 64 GiB, so this runs for a while. It can be interrupted and resumed.
    Download the archives manually from the links in https://github.com/funalab/FL2-Net and place
    them in `path` as 'raw.tar.gz' and 'gt.tar.gz' if the download fails.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder that holds the archives.
    """
    os.makedirs(path, exist_ok=True)

    for key, archive_name in ARCHIVE_NAMES.items():
        archive_path = os.path.join(path, archive_name)
        if os.path.exists(archive_path):
            continue
        if not download:
            raise RuntimeError(f"Cannot find the data at {archive_path}, but download was set to False")

        _download_from_gdrive(
            path=archive_path,
            file_id=FILE_IDS[key],
            total=ARCHIVE_SIZES[key],
            checksum=CHECKSUMS[key],
            desc=f"Download {archive_name}",
        )

        # The download only ever writes ranges that Google served as file content, but a corrupt
        # archive would otherwise not surface until the extraction fails with a confusing error.
        if not tarfile.is_tarfile(archive_path):
            os.remove(archive_path)
            raise RuntimeError(MANUAL_DOWNLOAD_MESSAGE.format(name=archive_name, path=archive_path))

    return path


def get_fl2net_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"] = "train",
    embryos: Optional[Sequence[str]] = None,
    timepoints: Optional[Sequence[int]] = None,
    stride: int = 25,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the FL2-Net data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. Either 'train', 'val' or 'test'.
        embryos: The embryos to use, for example 'F001/Embryo01'. Defaults to all of the split.
        timepoints: The timepoints to use, counted from one. Overrides `stride`.
        stride: The number of timepoints to skip between two extractions.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Choose from {list(SPLITS)}.")

    if embryos is None:
        embryos = SPLITS[split]
    else:
        for embryo in embryos:
            if embryo not in SPLITS[split]:
                raise ValueError(f"The embryo '{embryo}' is not part of the '{split}' split.")

    get_fl2net_data(path, download)
    selected = _get_timepoints(timepoints, stride)
    # The names are sorted here, so that the images and the labels stay paired up below.
    relative_names = [f"{embryo}/{timepoint:03d}.tif" for embryo in sorted(embryos) for timepoint in selected]

    image_dir = os.path.join(path, "images")
    label_dir = os.path.join(path, "labels")
    _extract_members(os.path.join(path, ARCHIVE_NAMES["images"]), relative_names, image_dir)
    _extract_members(os.path.join(path, ARCHIVE_NAMES["labels"]), relative_names, label_dir)

    image_paths = [os.path.join(image_dir, name) for name in relative_names]
    label_paths = [os.path.join(label_dir, name) for name in relative_names]
    assert len(image_paths) == len(label_paths)
    return image_paths, label_paths


def get_fl2net_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    split: Literal["train", "val", "test"] = "train",
    embryos: Optional[Sequence[str]] = None,
    timepoints: Optional[Sequence[int]] = None,
    stride: int = 25,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the FL2-Net dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The 3D patch shape to use for training.
        split: The data split. Either 'train', 'val' or 'test'.
        embryos: The embryos to use, for example 'F001/Embryo01'. Defaults to all of the split.
        timepoints: The timepoints to use, counted from one. Overrides `stride`.
        stride: The number of timepoints to skip between two extractions.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if len(patch_shape) != 3:
        raise ValueError(f"The FL2-Net patch shape must be three-dimensional, got {patch_shape}.")

    image_paths, label_paths = get_fl2net_paths(path, split, embryos, timepoints, stride, download)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, offsets=offsets, boundaries=boundaries, binary=binary,
    )
    kwargs = util.ensure_transforms(ndim=3, **kwargs)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        ndim=3,
        **kwargs,
    )


def get_fl2net_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    split: Literal["train", "val", "test"] = "train",
    embryos: Optional[Sequence[str]] = None,
    timepoints: Optional[Sequence[int]] = None,
    stride: int = 25,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the FL2-Net dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The 3D patch shape to use for training.
        split: The data split. Either 'train', 'val' or 'test'.
        embryos: The embryos to use, for example 'F001/Embryo01'. Defaults to all of the split.
        timepoints: The timepoints to use, counted from one. Overrides `stride`.
        stride: The number of timepoints to skip between two extractions.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_fl2net_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        embryos=embryos,
        timepoints=timepoints,
        stride=stride,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
