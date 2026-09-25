"""The Spine Endoscopic Atlas (SEA) dataset contains annotations for surgical instrument
segmentation in endoscopic spine surgery images.

The full archive ships 48,510 images in total, of which 10,662 come with an instrument
segmentation mask (stored as an NRRD file in a sibling 'seg' folder, matched to its image by
basename); the remaining ~37,848 images are unlabeled raw frames under 'unclassified' and are
not used by this loader. Annotated images are organized by working-channel diameter ('big' or
'small'), spinal region ('cervical' or 'lumbar'), patient, difficulty ('normal' or 'difficult'
scenario) and instrument type ('bipolar', 'grasping_forceps', 'drill', 'dissector', 'punch' or
'scissor'); the instrument type can be selected with the 'instrument' argument. This loader
discovers the image-mask pairs on disk rather than relying on the advertised counts, and merges
each NRRD segmentation (which may encode either a binary 0/255 mask or a per-segment integer
label map, depending on the file) into a binary instrument mask during preprocessing.

The dataset is located at https://doi.org/10.6084/m9.figshare.27109312, released under a CC0 license.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Optional

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://ndownloader.figshare.com/files/56202515"
CHECKSUM = "bde2014aa786181140552ab248fad0b2"

INSTRUMENTS = ["bipolar", "grasping_forceps", "drill", "dissector", "punch", "scissor"]


def _convert_mask(nrrd_path, out_path):
    import numpy as np
    import SimpleITK as sitk
    import tifffile

    if os.path.exists(out_path):
        return

    data = sitk.GetArrayFromImage(sitk.ReadImage(nrrd_path))
    data = np.squeeze(data)
    mask = (data > 0).astype("uint8")

    tmp_path = f"{out_path}.{os.getpid()}.incomplete.tif"
    tifffile.imwrite(tmp_path, mask)
    os.replace(tmp_path, out_path)


def _convert_raw(raw_path, out_path):
    from PIL import Image

    if os.path.exists(out_path):
        return

    image = Image.open(raw_path).convert("RGB")

    tmp_path = f"{out_path}.{os.getpid()}.incomplete.png"
    image.save(tmp_path)
    os.replace(tmp_path, out_path)


def get_spine_endoscopic_atlas_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Spine Endoscopic Atlas dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "Spine endoscopic atlas", "classified")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "spine_endoscopic_atlas.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path, remove=False)

    assert os.path.exists(data_dir), f"The extraction of the archive did not create the expected folder in '{path}'."

    return data_dir


def get_spine_endoscopic_atlas_paths(
    path: Union[os.PathLike, str], instrument: Optional[str] = None, download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the Spine Endoscopic Atlas data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        instrument: The choice of instrument type to restrict the data to. By default all instrument
            types are used. See `INSTRUMENTS` for the valid choices.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if instrument is not None and instrument not in INSTRUMENTS:
        raise ValueError(f"'{instrument}' is not a valid instrument. Choose one of {INSTRUMENTS}.")

    data_dir = get_spine_endoscopic_atlas_data(path, download)
    mask_dir = os.path.join(path, "spine_endoscopic_atlas_masks")
    raw_dir = os.path.join(path, "spine_endoscopic_atlas_raw")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    pattern = os.path.join(data_dir, "*", "*", "*", "*", instrument or "*", "*.*")
    raw_paths, label_paths = [], []
    for source_path in natsorted(glob(pattern)):
        if source_path.lower().endswith((".jpg", ".png")) and os.path.basename(os.path.dirname(source_path)) != "seg":
            base = os.path.splitext(os.path.basename(source_path))[0]
            nrrd_path = os.path.join(os.path.dirname(source_path), "seg", f"{base}.nrrd")
            if not os.path.exists(nrrd_path):
                continue

            # The raw images are re-encoded as RGB PNGs: some ship as RGBA, which breaks the
            # fixed 3-channel assumption of the resize transform used by 'resize_inputs'.
            raw_path = os.path.join(raw_dir, f"{base}.png")
            _convert_raw(source_path, raw_path)

            label_path = os.path.join(mask_dir, f"{base}.tif")
            _convert_mask(nrrd_path, label_path)

            raw_paths.append(raw_path)
            label_paths.append(label_path)

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    return raw_paths, label_paths


def get_spine_endoscopic_atlas_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    instrument: Optional[str] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Spine Endoscopic Atlas dataset for surgical instrument segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        instrument: The choice of instrument type to restrict the data to.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_spine_endoscopic_atlas_paths(path, instrument, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_spine_endoscopic_atlas_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    instrument: Optional[str] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Spine Endoscopic Atlas dataloader for surgical instrument segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        instrument: The choice of instrument type to restrict the data to.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_spine_endoscopic_atlas_dataset(path, patch_shape, instrument, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
