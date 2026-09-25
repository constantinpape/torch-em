"""The Lumbar Spine US dataset contains annotations for lumbar bone surface segmentation
in paired handheld (HUS) and robot-assisted (RUS) ultrasound frames, acquired together with
ground-truth CT of the lumbar spine in 63 healthy volunteers. Out of these, 9 participants
have expert-annotated bone surface masks for 6091 ultrasound frames in total (2353 HUS and
3738 RUS frames, according to the publication below).

The dataset is located at https://doi.org/10.48804/3XPCAE.
This dataset is from the publication https://doi.org/10.1038/s41597-025-06047-9.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://rdr.kuleuven.be/api/access/datafile/"

# Each entry maps a "<participant>_<scan>" recording, hosted on the KU Leuven RDR (Dataverse)
# repository, to its (file id, sha256 checksum, probe). Scans tagged 'H' are handheld (HUS)
# acquisitions. Scans tagged 'R' and 'D' are robot-assisted (RUS) acquisitions: 'D' denotes the
# robotic scan types other than the 'Perpendicular' and 'along the spinous process' ones (see the
# publication's Data Records section for the scan type naming convention).
SCANS = {
    "URS08_H1": (226877, "5b68e478100e5c445bbcabb8df67e6ea2aa7438e7ef85ac9445f3c93f1ebab8e", "handheld"),
    "URS08_R2": (226871, "62a1c0abcab00b894009ea2f7a06cfecfbaf95421b896cfbf65492797fcc632b", "robotic"),
    "URS16_H3": (227333, "3cf4892554902c724a0008453369eda2986a41b38c87571e53f0373009b5753d", "handheld"),
    "URS16_R1": (227210, "23d0738403c5ccf6bf895b9b363e758011804cf1709fcdf59791dd041c8eabd9", "robotic"),
    "URS26_H3": (226883, "0fbf944aa9acadc2bcb244d4e1234fd7330ad776ef7ba8cae856d3b62e251f92", "handheld"),
    "URS26_R1": (226873, "ba52bef856bc34e4c91177e976ca79982f22feb058a40eac009cf8c87159dca8", "robotic"),
    "URS31_D2": (227069, "eb18d445d1612cca6f243d9a6dd71f0b53bc40827569b69daf1e08d17e51227a", "robotic"),
    "URS31_R2": (226964, "ac8a69f34b9c2eff28ded73e2ddb5b44b15ae9bec240ea1eef23a5335c587733", "robotic"),
    "URS36_D2": (226975, "ffdcfef013706d43bf03011efc121507dca4ab3adfa771e697697a65d27bb29a", "robotic"),
    "URS36_H2": (227136, "5e203254f081ec5559949aef601675fef49f50b1d8c307d738c2d57e76621a69", "handheld"),
    "URS40_H4": (227343, "67fadae809baca2478858fecd8074acbdb0493501f4b3772e78669a0ad6ded6a", "handheld"),
    "URS40_R1": (227007, "430ea76b2eefe038f3e77c006417e597a4c6522830c367461cb07c46b55611a6", "robotic"),
    "URS45_H2": (227396, "3fb13ff323ef7cb6e8fe519bdb24784a3eeddbe74a0938b9eb4c46911aace470", "handheld"),
    "URS45_R2": (227066, "bcb4e83340b0edeec6e6a061d1d455702e49e60894bfec60a5b0b2702fca1046", "robotic"),
    "URS51_D2": (226893, "8d2e687c769f4fac6ce22a3acc51c2bedf79636b56fbe0bc9ec8ff0d7bf2c0df", "robotic"),
    "URS51_R2": (226825, "ac89115922ee6bf54a5c5fbc1710755e8d2386b2fb69caad67efc1e9c1b28ded", "robotic"),
    "URS54_D1": (226951, "8e4b8f45a0a9dc149a9f78ba6eb88e7ef817fe0535815607c6db862ab8c7ebcb", "robotic"),
    "URS54_H3": (226958, "9037e202700de1ce3f181a3937b9bbd1e43499526545ae7388375f489d04ba97", "handheld"),
}


def _read_metaimage(path):
    """Read a MetaImage ('.mhd' + '.raw') frame, returning it as a numpy array. Supports zlib compressed
    data, as used by the raw files shipped with this dataset.
    """
    import SimpleITK as sitk
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def get_lumbar_spine_us_data(
    path: Union[os.PathLike, str], probe: Literal["handheld", "robotic", "all"] = "all", download: bool = False
) -> str:
    """Download the Lumbar Spine US dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        probe: The choice of ultrasound probe. Either 'handheld', 'robotic' or 'all'.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    os.makedirs(path, exist_ok=True)

    scans = [name for name, (_, _, p) in SCANS.items() if probe == "all" or p == probe]
    for name in scans:
        scan_dir = os.path.join(path, name)
        if os.path.exists(scan_dir):
            continue

        file_id, checksum, _ = SCANS[name]
        zip_path = os.path.join(path, f"{name}.zip")
        util.download_source(path=zip_path, url=f"{BASE_URL}{file_id}", download=download, checksum=checksum)
        util.unzip(zip_path=zip_path, dst=path)

    return path


def _convert_scans_to_tif(path, scans):
    converted_dir = os.path.join(path, "converted")
    os.makedirs(converted_dir, exist_ok=True)

    image_paths, label_paths = [], []
    for name in scans:
        label_mhds = natsorted(glob(os.path.join(path, name, "Labels", "*-labels.mhd")))
        for label_mhd in tqdm(label_mhds, desc=f"Converting '{name}' to tif", leave=False):
            frame_id = os.path.basename(label_mhd).replace("-labels.mhd", "")
            raw_mhd = os.path.join(path, name, "Labels", f"{frame_id}.mhd")

            image_path = os.path.join(converted_dir, f"{name}_{frame_id}.tif")
            label_path = os.path.join(converted_dir, f"{name}_{frame_id}_labels.tif")
            image_paths.append(image_path)
            label_paths.append(label_path)
            if os.path.exists(image_path) and os.path.exists(label_path):
                continue

            image = _read_metaimage(raw_mhd)
            labels = _read_metaimage(label_mhd).astype("uint8")
            imageio.imwrite(image_path, image)
            imageio.imwrite(label_path, labels)

    return image_paths, label_paths


def get_lumbar_spine_us_paths(
    path: Union[os.PathLike, str], probe: Literal["handheld", "robotic", "all"] = "all", download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the Lumbar Spine US data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        probe: The choice of ultrasound probe. Either 'handheld', 'robotic' or 'all'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    get_lumbar_spine_us_data(path=path, probe=probe, download=download)

    scans = [name for name, (_, _, p) in SCANS.items() if probe == "all" or p == probe]
    image_paths, label_paths = _convert_scans_to_tif(path, scans)

    assert len(image_paths) == len(label_paths) and len(image_paths) > 0

    return image_paths, label_paths


def get_lumbar_spine_us_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    probe: Literal["handheld", "robotic", "all"] = "all",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Lumbar Spine US dataset for lumbar bone surface segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        probe: The choice of ultrasound probe. Either 'handheld', 'robotic' or 'all'.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_lumbar_spine_us_paths(path=path, probe=probe, download=download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
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
        **kwargs
    )


def get_lumbar_spine_us_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    probe: Literal["handheld", "robotic", "all"] = "all",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Lumbar Spine US dataloader for lumbar bone surface segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        probe: The choice of ultrasound probe. Either 'handheld', 'robotic' or 'all'.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_lumbar_spine_us_dataset(path, patch_shape, probe, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
