import os
from torch_em.data import ConcatDataset, PseudoLabelDataset
from torch_em.segmentation import (get_data_loader, get_raw_transform,
                                   is_segmentation_dataset,
                                   samples_to_datasets, _get_default_transform)
from .shallow2deep_model import Shallow2DeepModel


def check_paths(raw_paths):
    def _check_path(path):
        if not os.path.exists(path):
            raise ValueError(f"Could not find path {path}")

    if isinstance(raw_paths, str):
        _check_path(raw_paths)
    else:
        for rp in raw_paths:
            _check_path(rp)


def _load_pseudolabel_dataset(raw_paths, raw_key, **kwargs):
    rois = kwargs.pop("rois", None)
    if isinstance(raw_paths, str):
        if rois is not None:
            assert len(rois) == 3 and all(isinstance(roi, slice) for roi in rois)
        ds = PseudoLabelDataset(raw_paths, raw_key, roi=rois, labeler_device="cpu", **kwargs)
    else:
        assert len(raw_paths) > 0
        if rois is not None:
            assert len(rois) == len(raw_paths), f"{len(rois)}, {len(raw_paths)}"
            assert all(isinstance(roi, tuple) for roi in rois)
        n_samples = kwargs.pop("n_samples", None)

        samples_per_ds = (
            [None] * len(raw_paths) if n_samples is None else samples_to_datasets(n_samples, raw_paths, raw_key)
        )
        ds = []
        for i, raw_path in enumerate(raw_paths):
            roi = None if rois is None else rois[i]
            dset = PseudoLabelDataset(
                raw_path, raw_key, roi=roi, labeler_device="cpu", n_samples=samples_per_ds[i], **kwargs
            )
            ds.append(dset)
        ds = ConcatDataset(*ds)
    return ds


def get_pseudolabel_dataset(
    raw_paths,
    raw_key,
    checkpoint,
    rf_config,
    patch_shape,
    raw_transform=None,
    transform=None,
    rois=None,
    n_samples=None,
    ndim=None,
    is_raw_dataset=None,
    pseudo_labeler_device="cpu",
):
    check_paths(raw_paths)
    if is_raw_dataset is None:
        is_raw_dataset = is_segmentation_dataset(raw_paths, raw_key, raw_paths, raw_key)

    # we always use a raw transform in the convenience function
    if raw_transform is None:
        raw_transform = get_raw_transform()

    # we always use augmentations in the convenience function
    if transform is None:
        transform = _get_default_transform(
            raw_paths if isinstance(raw_paths, str) else raw_paths[0], raw_key, is_raw_dataset, ndim
        )

    pseudo_labeler = Shallow2DeepModel(checkpoint, rf_config, pseudo_labeler_device, ilastik_multi_thread=True) # TODO: does this work with multi thread?
    if is_raw_dataset:
        ds = _load_pseudolabel_dataset(
            raw_paths, raw_key,
            patch_shape=patch_shape,
            pseudo_labeler=pseudo_labeler,
            raw_transform=raw_transform,
            transform=transform,
            rois=rois, n_samples=n_samples, ndim=ndim,
        )
    else:
        raise NotImplementedError("Image collection dataset for shallow2deep not implemented yet.")
    return ds


# TODO add options for confidence module and consistency
def get_pseudolabel_loader(
    raw_paths,
    raw_key,
    checkpoint,
    rf_config,
    batch_size,
    patch_shape,
    raw_transform=None,
    transform=None,
    rois=None,
    n_samples=None,
    ndim=None,
    is_raw_dataset=None,
    pseudo_labeler_device="cpu",
    **loader_kwargs,
):
    ds = get_pseudolabel_dataset(
        raw_paths=raw_paths, raw_key=raw_key,
        checkpoint=checkpoint, rf_config=rf_config, patch_shape=patch_shape,
        raw_transform=raw_transform, transform=transform, rois=rois,
        n_samples=n_samples, ndim=ndim, is_raw_dataset=is_raw_dataset,
        pseudo_labeler_device=pseudo_labeler_device,
    )
    return get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
