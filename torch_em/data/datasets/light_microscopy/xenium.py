"""The Xenium datasets contain cell and nucleus segmentation masks for whole-slide
fluorescence images of human and mouse tissue, from the 10x Genomics in situ platform.

Every sample pairs a set of morphology images with the masks that the Xenium Onboard
Analysis (XOA) pipeline produced. Channel 0 always holds the DAPI nuclear stain. The
other channels hold the boundary and interior stains of the multi-tissue stain mix,
which XOA uses to draw the cell outlines.

NOTE: The masks are the output of XOA, not a manual annotation. XOA segments the nuclei
on DAPI, then grows each cell from its nucleus using the boundary and interior stains.
Where no stain supports a boundary it falls back to expanding the nucleus by 5 micrometer,
so a share of the cells are dilated nuclei rather than observed outlines. It runs between three
and five percent on the samples here. XOA reports that share from version 3.0 on, and each h5
file passes it through as the 'nuc_expansion_fraction' attribute, which stays 'nan' for the
earlier versions. Treat this data as a weak label source for pretraining, not as a benchmark.
For expert annotations on Xenium images see the SPATCH dataset in
`torch_em.data.datasets.histopathology.spatch`.

NOTE: A cell can hold more than one nucleus, so the two label sets do not match one to one
and their ids do not correspond.

NOTE: The layout of the bundle changed with the XOA version. Version 2 and 3 name the
morphology channels 'morphology_focus_000N.ome.tif' and version 4 names them
'ch000N_<stain>.ome.tif'. This loader sorts the files by name, which puts DAPI first in
both schemes, and stores the names it used in the 'channel_files' attribute. Versions
before 2.0 ship a different layout and this loader does not cover them.

NOTE: The images are whole slides that hold large empty regions. Pass a sampler such as
`torch_em.data.MinInstanceSampler` to avoid drawing empty patches.

The data is hosted at https://www.10xgenomics.com/datasets under the CC BY 4.0 license.
Please cite the corresponding dataset page if you use this data in your research.
"""

import os
import json
import zlib
import shutil
import struct
import hashlib
import zipfile
from glob import glob
from natsort import natsorted
from contextlib import contextmanager
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import requests

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


URLS = {
    "human_pancreas": "https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_human_Pancreas_FFPE/Xenium_V1_human_Pancreas_FFPE_xe_outs.zip",  # noqa
    "human_lung_cancer": "https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_humanLung_Cancer_FFPE/Xenium_V1_humanLung_Cancer_FFPE_xe_outs.zip",  # noqa
    "mouse_colon": "https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_mouse_Colon_FF/Xenium_V1_mouse_Colon_FF_xe_outs.zip",  # noqa
    "human_skin": "https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Human_Skin_FFPE/Xenium_Prime_Human_Skin_FFPE_xe_outs.zip",  # noqa
    "human_prostate": "https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Human_Prostate_FFPE/Xenium_Prime_Human_Prostate_FFPE_xe_outs.zip",  # noqa
    "human_breast": "https://cf.10xgenomics.com/samples/xenium/4.0.0/Human_Breast_Biomarkers_S1_Top/Human_Breast_Biomarkers_S1_Top_xe_outs.zip",  # noqa
}

# The sha256 of the two members that every download reads, keyed by the member name.
# The optional stain channels carry no entry here and rest on their crc32 alone.
CHECKSUMS = {
    "human_pancreas": {
        "cells.zarr.zip": "5747730a532f130b41f83e970784302623a784dfc9fe571dbd11e9179366387c",
        "morphology_focus/morphology_focus_0000.ome.tif":
            "3a7c669ebe6cbbd7e90ebab790bc8e666b9165e972bc528e134037eef632b3bd",
    },
    "human_lung_cancer": {
        "cells.zarr.zip": "f274590902f9162295d292e540b631ba37c03574a8db754a5c27f101c8c94ef1",
        "morphology_focus/morphology_focus_0000.ome.tif":
            "eb0ef6aa2c7adfa5e70323b4b9530771f8679bd9c0eb6dbd1af40faabc2ca398",
    },
    "mouse_colon": {
        "cells.zarr.zip": "cc5f811511d349955b15b6e4ebb4205a4208418f900ada5853231918d929a17e",
        "morphology_focus/morphology_focus_0000.ome.tif":
            "2a723a59b01c943d98a31aa47fc0297e6e9414b391c1d4d3949df0b27e510ef2",
    },
    "human_skin": {
        "cells.zarr.zip": "ef805ec5c7e733f7c82a0e88652a04550abfd9f7a6c60c52339aa34ff780316d",
        "morphology_focus/morphology_focus_0000.ome.tif":
            "435caf0d766d6226ab8a8c3be551c9a87ad2be1997d37767c992aa36bd7d09f3",
    },
    "human_prostate": {
        "cells.zarr.zip": "7ce2e92b1b085b35051f8568a983c8f8a8622e73bfae6060efdf7b47de6371e4",
        "morphology_focus/morphology_focus_0000.ome.tif":
            "a095080cc946f734545db51300d7ba24700c03b02cba48204e680c556dd867ed",
    },
    "human_breast": {
        "cells.zarr.zip": "61d09dd84509a09cf2f6c1252e7d12772f6b3d6f33dd02ea25555f8ad84b6758",
        "morphology_focus/ch0000_dapi.ome.tif":
            "75476f9be2b0f936022611d79717cb77356e90ec40902531098b984a54d9c4a0",
    },
}

# The tissue of every sample, for the h5 attributes.
TISSUES = {
    "human_pancreas": "human pancreas",
    "human_lung_cancer": "human lung adenocarcinoma",
    "mouse_colon": "mouse colon",
    "human_skin": "human skin melanoma",
    "human_prostate": "human prostate adenocarcinoma",
    "human_breast": "human breast",
}

LABEL_CHANNELS = {"nuclei": "0", "cells": "1"}

BLOCK = 4096


def _read_range(url: str, start: int, end: int) -> bytes:
    """Read the bytes from `start` to `end`, both included."""
    response = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True)
    response.raise_for_status()
    return response.content


def _read_zip64_extra(extra: bytes, uncompressed: int, compressed: int, offset: int) -> Tuple[int, int, int]:
    """Replace the fields that overflowed the 32 bit form with their zip64 values."""
    position = 0
    while position + 4 <= len(extra):
        tag, size = struct.unpack("<HH", extra[position:position + 4])
        if tag == 0x0001:
            values = list(struct.unpack(f"<{size // 8}Q", extra[position + 4:position + 4 + size]))
            if uncompressed == 0xFFFFFFFF:
                uncompressed = values.pop(0)
            if compressed == 0xFFFFFFFF:
                compressed = values.pop(0)
            if offset == 0xFFFFFFFF:
                offset = values.pop(0)
            break
        position += 4 + size
    return uncompressed, compressed, offset


def _remote_zip_index(url: str) -> Dict[str, Dict[str, int]]:
    """Read the central directory of a remote zip, which lists every member and where it sits.

    This turns the bundle into a set of files that can be read one by one, so that a download
    can skip the transcripts and the expression matrix, which segmentation never touches.
    """
    response = requests.head(url)
    response.raise_for_status()
    size = int(response.headers["Content-Length"])

    tail = _read_range(url, max(0, size - 262144), size - 1)
    position = tail.rfind(b"PK\x06\x06")
    if position == -1:
        position = tail.rfind(b"PK\x05\x06")
        if position == -1:
            raise RuntimeError(f"Could not find the central directory of the zip at {url}.")
        directory_size, directory_offset = struct.unpack("<II", tail[position + 12:position + 20])
    else:
        directory_size, directory_offset = struct.unpack("<QQ", tail[position + 40:position + 56])

    directory = _read_range(url, directory_offset, directory_offset + directory_size + 128)
    index, position = {}, 0
    while position + 46 <= len(directory) and directory[position:position + 4] == b"PK\x01\x02":
        method = struct.unpack("<H", directory[position + 10:position + 12])[0]
        crc = struct.unpack("<I", directory[position + 16:position + 20])[0]
        compressed, uncompressed = struct.unpack("<II", directory[position + 20:position + 28])
        name_length, extra_length, comment_length = struct.unpack("<HHH", directory[position + 28:position + 34])
        offset = struct.unpack("<I", directory[position + 42:position + 46])[0]
        name = directory[position + 46:position + 46 + name_length].decode("utf-8", errors="replace")
        extra = directory[position + 46 + name_length:position + 46 + name_length + extra_length]
        uncompressed, compressed, offset = _read_zip64_extra(extra, uncompressed, compressed, offset)
        index[name] = {"method": method, "crc": crc, "compressed": compressed, "offset": offset}
        position += 46 + name_length + extra_length + comment_length
    return index


def _download_member(url: str, name: str, entry: Dict[str, int], output_path: str, expected_sha: Optional[str]) -> None:
    """Read one member out of a remote zip and write it, checking its crc32 and its sha256."""
    from tqdm import tqdm

    # The local header repeats the name and carries its own extra field, so the data of the
    # member starts behind a header whose length only the header itself gives.
    header = _read_range(url, entry["offset"], entry["offset"] + 29)
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"The member '{name}' does not start with a local header.")
    name_length, extra_length = struct.unpack("<HH", header[26:30])
    start = entry["offset"] + 30 + name_length + extra_length

    if entry["method"] not in (0, 8):
        raise RuntimeError(f"The member '{name}' uses the unsupported compression method {entry['method']}.")
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS) if entry["method"] == 8 else None

    crc, digest = 0, hashlib.sha256()
    temporary_path = f"{output_path}.incomplete"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    headers = {"Range": f"bytes={start}-{start + entry['compressed'] - 1}"}
    with requests.get(url, headers=headers, stream=True) as response:
        response.raise_for_status()
        description = f"Download {name}"
        with open(temporary_path, "wb") as f:
            with tqdm(total=entry["compressed"], unit="B", unit_scale=True, desc=description) as progress:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    progress.update(len(chunk))
                    block = decompressor.decompress(chunk) if decompressor is not None else chunk
                    crc = zlib.crc32(block, crc)
                    digest.update(block)
                    f.write(block)
            if decompressor is not None:
                block = decompressor.flush()
                crc = zlib.crc32(block, crc)
                digest.update(block)
                f.write(block)

    if crc != entry["crc"]:
        os.remove(temporary_path)
        raise RuntimeError(f"The member '{name}' has the crc32 {crc}, but the zip lists {entry['crc']}.")
    if expected_sha is not None and digest.hexdigest() != expected_sha:
        os.remove(temporary_path)
        raise RuntimeError(f"The member '{name}' has the sha256 {digest.hexdigest()}, expected {expected_sha}.")
    os.replace(temporary_path, output_path)


def _download_bundle(url: str, bundle_dir: str, sample: str, download: bool, with_stains: bool) -> str:
    """Download the members that this loader reads, and skip the rest of the bundle."""
    if os.path.exists(bundle_dir):
        return bundle_dir
    if not download:
        raise RuntimeError(f"Cannot find the data at {bundle_dir}, but download was set to False")

    index = _remote_zip_index(url)
    channels = natsorted(n for n in index if n.startswith("morphology_focus/") and n.endswith(".ome.tif"))
    if not channels:
        raise RuntimeError(f"The bundle for '{sample}' holds no morphology channel.")
    if not with_stains:
        channels = channels[:1]

    names = ["experiment.xenium", "cells.zarr.zip"] + channels
    missing = [n for n in names if n not in index]
    if missing:
        raise RuntimeError(f"The bundle for '{sample}' is missing {missing}.")

    temporary_dir = f"{bundle_dir}.tmp"
    if os.path.exists(temporary_dir):
        shutil.rmtree(temporary_dir)
    for name in names:
        _download_member(url, name, index[name], os.path.join(temporary_dir, name), CHECKSUMS[sample].get(name))

    os.replace(temporary_dir, bundle_dir)
    return bundle_dir


def _channel_paths(bundle_dir: str) -> List[str]:
    """Return the morphology channels, DAPI first. Both naming schemes sort DAPI to the front."""
    paths = natsorted(glob(os.path.join(bundle_dir, "morphology_focus", "*.ome.tif")))
    if not paths:
        raise RuntimeError(f"Could not find any morphology channel in {bundle_dir}.")
    return paths


@contextmanager
def _open_image(path: str):
    """Open one morphology channel at full resolution, without reading it into memory.

    The channels form a multi file OME pyramid, which the OME reader of tifffile rejects.
    Reading the file on its own gives the pyramid of this channel alone.
    """
    import zarr
    import tifffile

    with tifffile.TiffFile(path, is_ome=False) as tif:
        store = tif.series[0].levels[0].aszarr()
        try:
            yield zarr.open(store, mode="r")["0"]
        finally:
            store.close()


def _copy_blockwise(source, target, channel: Optional[int] = None) -> int:
    """Copy a large 2d array block by block, to keep the memory use bounded.

    The target must be indexed in one step, because indexing an h5 dataset twice writes
    into a copy of the data rather than into the file.

    Returns the largest value that was copied, which gives the instance count of a mask
    without a second pass over the whole array.
    """
    height, width = source.shape[-2:]
    largest = 0
    for y in range(0, height, BLOCK):
        for x in range(0, width, BLOCK):
            box = (slice(y, min(y + BLOCK, height)), slice(x, min(x + BLOCK, width)))
            block = source[box]
            if channel is None:
                target[box] = block
            else:
                target[(channel,) + box] = block
            largest = max(largest, int(block.max()))
    return largest


def _create_h5(bundle_dir: str, output_path: str, sample: str, with_stains: bool) -> str:
    import h5py
    import zarr
    from tqdm import tqdm

    if os.path.exists(output_path):
        return output_path

    with open(os.path.join(bundle_dir, "experiment.xenium")) as f:
        manifest = json.load(f)

    masks_zip = os.path.join(bundle_dir, "cells.zarr.zip")
    masks_dir = os.path.join(bundle_dir, "cells.zarr")
    if not os.path.exists(masks_dir):
        with zipfile.ZipFile(masks_zip) as archive:
            archive.extractall(masks_dir)
    masks = zarr.open(masks_dir, mode="r")["masks"]

    channel_paths = _channel_paths(bundle_dir)
    with _open_image(channel_paths[0]) as dapi:
        shape = dapi.shape
    for name, key in LABEL_CHANNELS.items():
        if masks[key].shape != shape:
            raise RuntimeError(
                f"The {name} mask of '{sample}' has the shape {masks[key].shape}, "
                f"but its DAPI image has the shape {shape}."
            )

    temporary_path = f"{output_path}.tmp"
    with h5py.File(temporary_path, "w") as f:
        f.attrs["sample"] = sample
        f.attrs["tissue"] = TISSUES[sample]
        f.attrs["axes"] = "yx"
        f.attrs["pixel_size"] = manifest["pixel_size"]
        f.attrs["xoa_version"] = manifest["analysis_sw_version"]
        f.attrs["segmentation_stain"] = manifest.get("segmentation_stain", "none")
        f.attrs["nuc_expansion_fraction"] = manifest.get("segmented_cell_nuc_expansion_frac", float("nan"))
        f.attrs["num_cells"] = manifest["num_cells"]
        f.attrs["channel_files"] = [os.path.basename(p) for p in channel_paths]

        chunks = (min(1024, shape[0]), min(1024, shape[1]))
        raw = f.create_dataset("raw/dapi", shape=shape, dtype="uint16", chunks=chunks, compression="gzip")
        raw.attrs["pixel_size"] = manifest["pixel_size"]
        with _open_image(channel_paths[0]) as dapi:
            _copy_blockwise(dapi, raw)

        if with_stains:
            stack_shape = (len(channel_paths),) + shape
            stack = f.create_dataset(
                "raw/stack", shape=stack_shape, dtype="uint16", chunks=(1,) + chunks, compression="gzip"
            )
            stack.attrs["pixel_size"] = manifest["pixel_size"]
            for channel, channel_path in enumerate(tqdm(channel_paths, desc=f"Copy channels of '{sample}'")):
                with _open_image(channel_path) as image:
                    if image.shape != shape:
                        raise RuntimeError(
                            f"The channel {os.path.basename(channel_path)} of '{sample}' has the shape "
                            f"{image.shape}, but its DAPI image has the shape {shape}."
                        )
                    _copy_blockwise(image, stack, channel=channel)

        for name, key in LABEL_CHANNELS.items():
            labels = f.create_dataset(
                f"labels/{name}", shape=shape, dtype="uint32", chunks=chunks, compression="gzip"
            )
            labels.attrs["pixel_size"] = manifest["pixel_size"]
            labels.attrs["num_instances"] = _copy_blockwise(masks[key], labels)

    os.replace(temporary_path, output_path)
    shutil.rmtree(masks_dir, ignore_errors=True)
    return output_path


def get_xenium_data(
    path: Union[os.PathLike, str],
    sample: str,
    download: bool = False,
    with_stains: bool = True,
) -> str:
    """Download one Xenium sample.

    The source is the Xenium Explorer subset of the output bundle, which holds the masks and
    the morphology images but not the transcripts. This function does not fetch the whole
    archive. It reads the central directory of the remote zip, then reads only the members
    that it needs, which cuts the download by about four times without the stains.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        sample: The sample to use. See `URLS` for the available samples.
        download: Whether to download the data if it is not present.
        with_stains: Whether to also store the boundary and interior stains next to DAPI.
            The cell masks rest on those stains, so a model that predicts cells needs them.

    Returns:
        The filepath to the h5 file of the sample.
    """
    if sample not in URLS:
        raise ValueError(f"'{sample}' is not a valid sample. Choose from {list(URLS)}.")

    output_path = os.path.join(path, "preprocessed", f"{sample}.h5")
    if os.path.exists(output_path):
        return output_path

    os.makedirs(os.path.join(path, "preprocessed"), exist_ok=True)
    bundle_dir = _download_bundle(URLS[sample], os.path.join(path, "bundles", sample), sample, download, with_stains)
    _create_h5(bundle_dir, output_path, sample, with_stains)
    shutil.rmtree(bundle_dir, ignore_errors=True)

    return output_path


def get_xenium_paths(
    path: Union[os.PathLike, str],
    sample: Optional[Union[str, Sequence[str]]] = None,
    download: bool = False,
    with_stains: bool = True,
) -> List[str]:
    """Get paths to the Xenium data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        sample: The sample or samples to use. Defaults to all of them.
        download: Whether to download the data if it is not present.
        with_stains: Whether to also store the boundary and interior stains next to DAPI.

    Returns:
        List of filepaths for the h5 data.
    """
    if sample is None:
        samples = list(URLS)
    else:
        samples = [sample] if isinstance(sample, str) else list(sample)

    return [get_xenium_data(path, name, download, with_stains) for name in samples]


def get_xenium_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    sample: Optional[Union[str, Sequence[str]]] = None,
    label_channel: Literal["nuclei", "cells"] = "nuclei",
    raw_channel: Literal["dapi", "stack"] = "dapi",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the Xenium dataset for cell and nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The 2D patch shape to use for training.
        sample: The sample or samples to use. Defaults to all of them.
        label_channel: The masks to use as target. Either 'nuclei' or 'cells'.
        raw_channel: The images to use as input. Either 'dapi' for the nuclear stain alone,
            or 'stack' for all morphology channels.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if len(patch_shape) != 2:
        raise ValueError(f"The Xenium patch shape must be two-dimensional, got {patch_shape}.")
    if label_channel not in LABEL_CHANNELS:
        raise ValueError(f"'{label_channel}' is not a valid label channel. Choose from {list(LABEL_CHANNELS)}.")
    if raw_channel not in ("dapi", "stack"):
        raise ValueError(f"'{raw_channel}' is not a valid raw channel. Choose from ['dapi', 'stack'].")

    volume_paths = get_xenium_paths(path, sample, download, with_stains=raw_channel == "stack")

    kwargs = util.update_kwargs(kwargs, "with_channels", raw_channel == "stack")
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, offsets=offsets, boundaries=boundaries, binary=binary,
    )
    kwargs = util.ensure_transforms(ndim=2, **kwargs)

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key=f"raw/{raw_channel}",
        label_paths=volume_paths,
        label_key=f"labels/{label_channel}",
        patch_shape=patch_shape,
        ndim=2,
        **kwargs,
    )


def get_xenium_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    sample: Optional[Union[str, Sequence[str]]] = None,
    label_channel: Literal["nuclei", "cells"] = "nuclei",
    raw_channel: Literal["dapi", "stack"] = "dapi",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the Xenium dataloader for cell and nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The 2D patch shape to use for training.
        sample: The sample or samples to use. Defaults to all of them.
        label_channel: The masks to use as target. Either 'nuclei' or 'cells'.
        raw_channel: The images to use as input. Either 'dapi' or 'stack'.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_xenium_dataset(
        path=path,
        patch_shape=patch_shape,
        sample=sample,
        label_channel=label_channel,
        raw_channel=raw_channel,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
