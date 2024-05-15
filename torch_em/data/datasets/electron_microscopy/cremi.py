"""CREMI is a dataset for neuron segmentation in EM.

It contains three annotated volumes from the adult fruit-fly brain.
It was held as a challenge at MICCAI 2016. For details on the dataset check out https://cremi.org/.
"""
# TODO add support for realigned volumes

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch_em
from torch.utils.data import Dataset, DataLoader

from .. import util

CREMI_URLS = {
    "original": {
        "A": "https://cremi.org/static/data/sample_A_20160501.hdf",
        "B": "https://cremi.org/static/data/sample_B_20160501.hdf",
        "C": "https://cremi.org/static/data/sample_C_20160501.hdf",
    },
    "realigned": {},
    "defects": "https://zenodo.org/record/5767036/files/sample_ABC_padded_defects.h5"
}
CHECKSUMS = {
    "original": {
        "A": "4c563d1b78acb2bcfb3ea958b6fe1533422f7f4a19f3e05b600bfa11430b510d",
        "B": "887e85521e00deead18c94a21ad71f278d88a5214c7edeed943130a1f4bb48b8",
        "C": "2874496f224d222ebc29d0e4753e8c458093e1d37bc53acd1b69b19ed1ae7052",
    },
    "realigned": {},
    "defects": "7b06ffa34733b2c32956ea5005e0cf345e7d3a27477f42f7c905701cdc947bd0"
}


def get_cremi_data(
    path: Union[os.PathLike, str],
    samples: Tuple[str],
    download: bool,
    use_realigned: bool = False,
) -> List[str]:
    """Download the CREMI training data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        samples: The CREMI samples to use. The available samples are 'A', 'B', 'C'.
        download: Whether to download the data if it is not present.
        use_realigned: Use the realigned instead of the original training data.

    Returns:
        The filepaths to the training data.
    """
    if use_realigned:
        # we need to sample batches in this case
        # sampler = torch_em.data.MinForegroundSampler(min_fraction=0.05, p_reject=.75)
        raise NotImplementedError
    else:
        urls = CREMI_URLS["original"]
        checksums = CHECKSUMS["original"]

    os.makedirs(path, exist_ok=True)
    data_paths = []
    for name in samples:
        url = urls[name]
        checksum = checksums[name]
        data_path = os.path.join(path, f"sample{name}.h5")
        # CREMI SSL certificates expired, so we need to disable verification
        util.download_source(data_path, url, download, checksum, verify=False)
        data_paths.append(data_path)
    return data_paths


def get_cremi_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    samples: Tuple[str, ...] = ("A", "B", "C"),
    use_realigned: bool = False,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    rois: Dict[str, Any] = {},
    defect_augmentation_kwargs: Dict[str, Any] = {
        "p_drop_slice": 0.025,
        "p_low_contrast": 0.025,
        "p_deform_slice": 0.0,
        "deformation_mode": "compress",
    },
    **kwargs,
) -> Dataset:
    """Get the CREMI dataset for the segmentation of neurons in EM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        samples: The CREMI samples to use. The available samples are 'A', 'B', 'C'.
        use_realigned: Use the realigned instead of the original training data.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        rois: The region of interests to use for the samples.
        defect_augmentation_kwargs: Keyword arguments for defect augmentations.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
       The segmentation dataset.
    """
    assert len(patch_shape) == 3
    if rois is not None:
        assert isinstance(rois, dict)

    data_paths = get_cremi_data(path, samples, download, use_realigned)
    data_rois = [rois.get(name, np.s_[:, :, :]) for name in samples]

    if defect_augmentation_kwargs is not None and "artifact_source" not in defect_augmentation_kwargs:
        # download the defect volume
        url = CREMI_URLS["defects"]
        checksum = CHECKSUMS["defects"]
        defect_path = os.path.join(path, "cremi_defects.h5")
        util.download_source(defect_path, url, download, checksum)
        defect_patch_shape = (1,) + tuple(patch_shape[1:])
        artifact_source = torch_em.transform.get_artifact_source(defect_path, defect_patch_shape,
                                                                 min_mask_fraction=0.75,
                                                                 raw_key="defect_sections/raw",
                                                                 mask_key="defect_sections/mask")
        defect_augmentation_kwargs.update({"artifact_source": artifact_source})

    raw_key = "volumes/raw"
    label_key = "volumes/labels/neuron_ids"

    # defect augmentations
    if defect_augmentation_kwargs is not None:
        raw_transform = torch_em.transform.get_raw_transform(
            augmentation1=torch_em.transform.EMDefectAugmentation(**defect_augmentation_kwargs)
        )
        kwargs = util.update_kwargs(kwargs, "raw_transform", raw_transform)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=False, boundaries=boundaries, offsets=offsets
    )

    return torch_em.default_segmentation_dataset(
        data_paths, raw_key, data_paths, label_key, patch_shape, rois=data_rois, **kwargs
    )


def get_cremi_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    samples: Tuple[str, ...] = ("A", "B", "C"),
    use_realigned: bool = False,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    rois: Dict[str, Any] = {},
    defect_augmentation_kwargs: Dict[str, Any] = {
        "p_drop_slice": 0.025,
        "p_low_contrast": 0.025,
        "p_deform_slice": 0.0,
        "deformation_mode": "compress",
    },
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for EM neuron segmentation in the CREMI dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        samples: The CREMI samples to use. The available samples are 'A', 'B', 'C'.
        use_realigned: Use the realigned instead of the original training data.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        rois: The region of interests to use for the samples.
        defect_augmentation_kwargs: Keyword arguments for defect augmentations.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    dataset_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    ds = get_cremi_dataset(
        path=path,
        patch_shape=patch_shape,
        samples=samples,
        use_realigned=use_realigned,
        download=download,
        offsets=offsets,
        boundaries=boundaries,
        rois=rois,
        defect_augmentation_kwargs=defect_augmentation_kwargs,
        **dataset_kwargs,
    )
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
