import os
from typing import Tuple

import numpy as np
from torch.utils.data import Dataset

import torch_em
from .util import download_source, update_kwargs

CREMI_URLS = {
    "original": {
        "A": "https://cremi.org/static/data/sample_A_20160501.hdf",
        "B": "https://cremi.org/static/data/sample_B_20160501.hdf",
        "C": "https://cremi.org/static/data/sample_C_20160501.hdf",
    },
    "realigned": {},
}
CHECKSUMS = {
    "original": {
        "A": "4c563d1b78acb2bcfb3ea958b6fe1533422f7f4a19f3e05b600bfa11430b510d",
        "B": "887e85521e00deead18c94a21ad71f278d88a5214c7edeed943130a1f4bb48b8",
        "C": "2874496f224d222ebc29d0e4753e8c458093e1d37bc53acd1b69b19ed1ae7052",
    },
    "realigned": {},
}


# TODO add support for realigned volumes
def get_cremi_loader(
    path,
    patch_shape,
    samples=("A", "B", "C"),
    use_realigned=False,
    download=False,
    offsets=None,
    boundaries=False,
    rois={},
    defect_augmentation_kwargs={
        "p_drop_slice": 0.025,
        "p_low_contrast": 0.025,
        "p_deform_slice": 0.0,
        "deformation_mode": "compress",
    },
    batch_size=1,
    loader_kwargs=None,
    **dataset_kwargs,
):
    """
    """
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
    return torch_em.get_data_loader(ds, batch_size=batch_size, **(loader_kwargs or {}))


def get_cremi_dataset(
    path,
    patch_shape,
    samples=("A", "B", "C"),
    use_realigned=False,
    download=False,
    offsets=None,
    boundaries=False,
    rois={},
    defect_augmentation_kwargs={
        "p_drop_slice": 0.025,
        "p_low_contrast": 0.025,
        "p_deform_slice": 0.0,
        "deformation_mode": "compress",
    },
    **kwargs,
) -> Tuple[Dataset, dict]:
    assert len(patch_shape) == 3
    if rois is not None:
        assert isinstance(rois, dict)
    os.makedirs(path, exist_ok=True)

    if use_realigned:
        # we need to sample batches in this case
        # sampler = torch_em.data.MinForegroundSampler(min_fraction=0.05, p_reject=.75)
        raise NotImplementedError
    else:
        urls = CREMI_URLS["original"]
        checksums = CHECKSUMS["original"]

    data_paths = []
    data_rois = []
    for name in samples:
        url = urls[name]
        checksum = checksums[name]
        data_path = os.path.join(path, f"sample{name}.h5")
        # CREMI SSL certificates expired, so we need to disable verification
        download_source(data_path, url, download, checksum, verify=False)
        data_paths.append(data_path)
        data_rois.append(rois.get(name, np.s_[:, :, :]))

    kwargs = update_kwargs(kwargs, "patch_shape", patch_shape)
    kwargs = update_kwargs(kwargs, "ndim", 3)
    kwargs = update_kwargs(kwargs, "rois", data_rois)

    raw_key = "volumes/raw"
    label_key = "volumes/labels/neuron_ids"

    # defect augmentations
    raw_transform = torch_em.transform.get_raw_transform(
        augmentation1=torch_em.transform.EMDefectAugmentation(**defect_augmentation_kwargs)
    )
    update_kwargs(kwargs, "raw_transform", raw_transform)

    assert not ((offsets is not None) and boundaries)
    if offsets is not None:
        # we add a binary target channel for foreground background segmentation
        label_transform = torch_em.transform.label.AffinityTransform(
            offsets=offsets, ignore_label=None, add_binary_target=False, add_mask=True
        )
        msg = "Offsets are passed, but 'label_transform2' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, "label_transform2", label_transform, msg=msg)
    elif boundaries:
        label_transform = torch_em.transform.label.BoundaryTransform()
        msg = "Boundaries is set to true, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, "label_transform", label_transform, msg=msg)

    return torch_em.default_segmentation_dataset(data_paths, raw_key, data_paths, label_key, **kwargs)
