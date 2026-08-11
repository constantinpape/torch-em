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
import shutil
import zipfile
from glob import glob
from natsort import natsorted
from contextlib import contextmanager
from typing import List, Literal, Optional, Sequence, Tuple, Union

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

CHECKSUMS = {
    "human_pancreas": "b4b1905083ebcd4dad2cbf18be939acef023d591a99b37402be9eeb62c2f1188",
    "human_lung_cancer": "ce56e1023afe4b1b5d54aebb8af6780f25f24cf7cc06fe9ac92df87a280555cc",
    "mouse_colon": "0f1439ac7cfb61ec9e34af942fe20a4352c647811b3d1849c5f351454b2a9370",
    "human_skin": "468173141e7e4f3c626e94583857fa5838e18ec8f52f5af7f51fd8066103fbaa",
    "human_prostate": "dafc459016ffc8ed5cb097959e3958c72c9e404d640b61d4feab4d3374aa9d7a",
    "human_breast": "b9fa810ff8a28864691835a5ee2aea11105679fabcaeb0de2ffd05b9f0b10ef7",
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

# The members of the bundle that this loader reads. The bundle also holds the transcripts
# and the expression matrix, which take most of its size and which segmentation does not need.
BUNDLE_MEMBERS = ("experiment.xenium", "cells.zarr.zip", "morphology_focus/")

BLOCK = 4096


def _extract_bundle(zip_path: str, bundle_dir: str, with_stains: bool) -> str:
    """Extract the images, the masks and the manifest, and leave the transcripts in the archive.

    Without the stains only the DAPI channel comes out, which saves about a gigabyte of
    transient disk space per sample.
    """
    if os.path.exists(bundle_dir):
        return bundle_dir

    temporary_dir = f"{bundle_dir}.tmp"
    if os.path.exists(temporary_dir):
        shutil.rmtree(temporary_dir)

    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        missing = [m for m in BUNDLE_MEMBERS if not any(n.startswith(m) for n in names)]
        if missing:
            raise RuntimeError(f"The bundle {os.path.basename(zip_path)} is missing {missing}.")

        channels = natsorted(n for n in names if n.startswith("morphology_focus/") and n.endswith(".ome.tif"))
        if not with_stains:
            channels = channels[:1]
        members = [n for n in names if n in ("experiment.xenium", "cells.zarr.zip")] + channels
        archive.extractall(temporary_dir, members=members)

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

    The loader reads the Xenium Explorer subset of the output bundle, which holds the masks
    and the morphology images but not the transcripts. It is around ten times smaller than
    the full bundle. The bundles of the six samples take 19 GB together.

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
    zip_path = os.path.join(path, "downloads", f"{sample}.zip")
    os.makedirs(os.path.join(path, "downloads"), exist_ok=True)
    util.download_source(zip_path, URLS[sample], download, CHECKSUMS[sample])

    bundle_dir = _extract_bundle(zip_path, os.path.join(path, "bundles", sample), with_stains)
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
