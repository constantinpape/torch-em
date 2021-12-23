import pickle

import numpy as np
import torch
from torch_em.segmentation import (check_paths, is_segmentation_dataset,
                                   get_data_loader, get_raw_transform,
                                   samples_to_datasets, _get_default_transform)
from torch_em.data import ConcatDataset, SegmentationDataset
from .prepare_shallow2deep import _get_filters, _apply_filters
from ..util import ensure_tensor_with_channels, ensure_spatial_array


class Shallow2DeepDataset(SegmentationDataset):
    _rf_paths = None
    _filter_config = None

    @property
    def rf_paths(self):
        return self._rf_paths

    @rf_paths.setter
    def rf_paths(self, value):
        self._rf_paths = value

    @property
    def filter_config(self):
        return self._filter_config

    @filter_config.setter
    def filter_config(self, value):
        self._filter_config = value

    def _predict_rf(self, raw):
        n_rfs = len(self._rf_paths)
        rf_path = self._rf_paths[np.random.randint(0, n_rfs)]
        with open(rf_path, "rb") as f:
            rf = pickle.load(f)
        filters_and_sigmas = _get_filters(self.ndim, self._filter_config)
        features = _apply_filters(raw, filters_and_sigmas)
        assert rf.n_features_in_ == features.shape[1]
        # NOTE: we always select the predictions for the foreground class here.
        # for multi-class training where we need multiple predictions this would need to be changed
        prediction = rf.predict_proba(features)[:, 1]
        prediction = prediction.reshape(raw.shape)
        return prediction

    def __getitem__(self, index):
        assert self._rf_paths is not None
        raw, labels = self._get_sample(index)
        initial_label_dtype = labels.dtype

        if self.raw_transform is not None:
            raw = self.raw_transform(raw)
        if self.label_transform is not None:
            labels = self.label_transform(labels)
        if self.transform is not None:
            raw, labels = self.transform(raw, labels)
            if self.trafo_halo is not None:
                raw = self.crop(raw)
                labels = self.crop(labels)
        # support enlarging bounding box here as well (for affinity transform) ?
        if self.label_transform2 is not None:
            labels = ensure_spatial_array(labels, self.ndim, dtype=initial_label_dtype)
            labels = self.label_transform2(labels)

        if isinstance(raw, (list, tuple)):  # this can be a list or tuple due to transforms
            assert len(raw) == 1
            raw = raw[0]
        raw = ensure_tensor_with_channels(raw, ndim=self._ndim, dtype=self.dtype)
        if raw.shape[0] > 1:
            raise NotImplementedError(
                f"Shallow2Deep training not implemented for multi-channel input yet; got {raw.shape[0]} channels"
            )

        prediction = self._predict_rf(raw[0].numpy())
        prediction = ensure_tensor_with_channels(prediction, ndim=self._ndim, dtype=self.dtype)
        return prediction, labels


def _load_shallow2deep_dataset(raw_paths, raw_key, label_paths, label_key, rf_paths, **kwargs):
    rois = kwargs.pop("rois", None)
    if isinstance(raw_paths, str):
        if rois is not None:
            assert len(rois) == 3 and all(isinstance(roi, slice) for roi in rois)
        ds = SegmentationDataset(raw_paths, raw_key, label_paths, label_key, roi=rois, **kwargs)
        ds.rf_paths = rf_paths
    else:
        assert len(raw_paths) > 0
        if rois is not None:
            assert len(rois) == len(label_paths)
            assert all(isinstance(roi, tuple) for roi in rois)
        n_samples = kwargs.pop("n_samples", None)

        samples_per_ds = (
            [None] * len(raw_paths) if n_samples is None else samples_to_datasets(n_samples, raw_paths, raw_key)
        )
        ds = []
        for i, (raw_path, label_path) in enumerate(zip(raw_paths, label_paths)):
            roi = None if rois is None else rois[i]
            dset = SegmentationDataset(
                raw_path, raw_key, label_path, label_key, roi=roi, n_samples=samples_per_ds[i], **kwargs
            )
            dset.rf_paths = rf_paths
            ds.append(dset)
        ds = ConcatDataset(*ds)
    return ds


def get_shallow2deep_dataset(
    raw_paths,
    raw_key,
    label_paths,
    label_key,
    rf_paths,
    patch_shape,
    raw_transform=None,
    label_transform=None,
    transform=None,
    dtype=torch.float32,
    rois=None,
    n_samples=None,
    sampler=None,
    ndim=None,
    is_seg_dataset=None,
    with_channels=False,
):
    check_paths(raw_paths, label_paths)
    if is_seg_dataset is None:
        is_seg_dataset = is_segmentation_dataset(raw_paths, raw_key,
                                                 label_paths, label_key)

    # we always use a raw transform in the convenience function
    if raw_transform is None:
        raw_transform = get_raw_transform()

    # we always use augmentations in the convenience function
    if transform is None:
        transform = _get_default_transform(
            raw_paths if isinstance(raw_paths, str) else raw_paths[0], raw_key, is_seg_dataset, ndim
        )

    if is_seg_dataset:
        ds = _load_shallow2deep_dataset(
            raw_paths,
            raw_key,
            label_paths,
            label_key,
            rf_paths,
            patch_shape=patch_shape,
            raw_transform=raw_transform,
            label_transform=label_transform,
            transform=transform,
            rois=rois,
            n_samples=n_samples,
            sampler=sampler,
            ndim=ndim,
            dtype=dtype,
            with_channels=with_channels,
        )
    else:
        raise NotImplementedError("Image collection dataset for shallow2deep not implemented yet.")
    return ds


def get_shallow2deep_loader(
    raw_paths,
    raw_key,
    label_paths,
    label_key,
    rf_paths,
    batch_size,
    patch_shape,
    filter_config=None,
    raw_transform=None,
    label_transform=None,
    transform=None,
    rois=None,
    n_samples=None,
    ndim=None,
    is_seg_dataset=None,
    with_channels=False,
    **loader_kwargs,
):
    ds = get_shallow2deep_dataset(
        raw_paths=raw_paths,
        raw_key=raw_key,
        label_paths=label_paths,
        label_key=label_key,
        rf_paths=rf_paths,
        patch_shape=patch_shape,
        raw_transform=raw_transform,
        label_transform=label_transform,
        transform=transform,
        rois=rois,
        n_samples=n_samples,
        ndim=ndim,
        is_seg_dataset=is_seg_dataset,
        with_channels=with_channels,
    )
    return get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
