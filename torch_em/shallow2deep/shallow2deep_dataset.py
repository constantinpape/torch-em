import os
import pickle
import warnings
from glob import glob

import numpy as np
import torch
from torch_em.segmentation import (check_paths, is_segmentation_dataset,
                                   get_data_loader, get_raw_transform,
                                   samples_to_datasets, _get_default_transform)
from torch_em.data import ConcatDataset, ImageCollectionDataset, SegmentationDataset
from .prepare_shallow2deep import _get_filters, _apply_filters
from ..util import ensure_tensor_with_channels, ensure_spatial_array


class _Shallow2DeepBase:
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

    @property
    def rf_channels(self):
        return self._rf_channels

    @rf_channels.setter
    def rf_channels(self, value):
        if isinstance(value, int):
            self.rf_channels = (value,)
        else:
            assert isinstance(value, tuple)
            self._rf_channels = value

    def _predict(self, raw, rf, filters_and_sigmas):
        features = _apply_filters(raw, filters_and_sigmas)
        assert rf.n_features_in_ == features.shape[1], f"{rf.n_features_in_}, {features.shape[1]}"

        try:
            pred_ = rf.predict_proba(features)
            assert pred_.shape[1] > max(self.rf_channels), f"{pred_.shape}, {self.rf_channels}"
            pred_ = pred_[:, self.rf_channels]
        except IndexError:
            warnings.warn(f"Random forest prediction failed for input features of shape: {features.shape}")
            pred_shape = (len(features), len(self.rf_channels))
            pred_ = np.zeros(pred_shape, dtype="float32")

        spatial_shape = raw.shape
        out_shape = (len(self.rf_channels),) + spatial_shape
        prediction = np.zeros(out_shape, dtype="float32")
        for chan in range(pred_.shape[1]):
            prediction[chan] = pred_[:, chan].reshape(spatial_shape)

        return prediction

    def _predict_rf(self, raw):
        n_rfs = len(self._rf_paths)
        rf_path = self._rf_paths[np.random.randint(0, n_rfs)]
        with open(rf_path, "rb") as f:
            rf = pickle.load(f)
        filters_and_sigmas = _get_filters(self.ndim, self._filter_config)
        return self._predict(raw, rf, filters_and_sigmas)

    def _predict_rf_anisotropic(self, raw):
        n_rfs = len(self._rf_paths)
        rf_path = self._rf_paths[np.random.randint(0, n_rfs)]
        with open(rf_path, "rb") as f:
            rf = pickle.load(f)
        filters_and_sigmas = _get_filters(2, self._filter_config)

        n_channels = len(self.rf_channels)
        prediction = np.zeros((n_channels,) + raw.shape, dtype="float32")
        for z in range(raw.shape[0]):
            pred = self._predict(raw[z], rf, filters_and_sigmas)
            prediction[:, z] = pred

        return prediction


class Shallow2DeepDataset(SegmentationDataset, _Shallow2DeepBase):
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

        # NOTE we assume single channel raw data here; this needs to be changed for multi-channel
        if getattr(self, "is_anisotropic", False):
            prediction = self._predict_rf_anisotropic(raw[0].numpy())
        else:
            prediction = self._predict_rf(raw[0].numpy())
        prediction = ensure_tensor_with_channels(prediction, ndim=self._ndim, dtype=self.dtype)
        labels = ensure_tensor_with_channels(labels, ndim=self._ndim, dtype=self.label_dtype)
        return prediction, labels


class Shallow2DeepImageCollectionDataset(ImageCollectionDataset, _Shallow2DeepBase):
    def __getitem__(self, index):
        raw, labels = self._get_sample(index)
        initial_label_dtype = labels.dtype

        if self.raw_transform is not None:
            raw = self.raw_transform(raw)

        if self.label_transform is not None:
            labels = self.label_transform(labels)

        if self.transform is not None:
            raw, labels = self.transform(raw, labels)

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

        # NOTE we assume single channel raw data here; this needs to be changed for multi-channel
        prediction = self._predict_rf(raw[0].numpy())
        prediction = ensure_tensor_with_channels(prediction, ndim=self._ndim, dtype=self.dtype)
        labels = ensure_tensor_with_channels(labels, ndim=self._ndim, dtype=self.label_dtype)
        return prediction, labels


def _load_shallow2deep_segmentation_dataset(
    raw_paths, raw_key, label_paths, label_key, rf_paths, rf_channels, ndim, **kwargs
):
    rois = kwargs.pop("rois", None)
    filter_config = kwargs.pop("filter_config", None)
    if ndim == "anisotropic":
        ndim = 3
        is_anisotropic = True
    else:
        is_anisotropic = False

    if isinstance(raw_paths, str):
        if rois is not None:
            assert len(rois) == 3 and all(isinstance(roi, slice) for roi in rois)
        ds = Shallow2DeepDataset(raw_paths, raw_key, label_paths, label_key, roi=rois, ndim=ndim, **kwargs)
        ds.rf_paths = rf_paths
        ds.filter_config = filter_config
        ds.rf_channels = rf_channels
        ds.is_anisotropic = is_anisotropic
    else:
        assert len(raw_paths) > 0
        if rois is not None:
            assert len(rois) == len(label_paths), f"{len(rois)}, {len(label_paths)}"
            assert all(isinstance(roi, tuple) for roi in rois)
        n_samples = kwargs.pop("n_samples", None)

        samples_per_ds = (
            [None] * len(raw_paths) if n_samples is None else samples_to_datasets(n_samples, raw_paths, raw_key)
        )
        ds = []
        for i, (raw_path, label_path) in enumerate(zip(raw_paths, label_paths)):
            roi = None if rois is None else rois[i]
            dset = Shallow2DeepDataset(
                raw_path, raw_key, label_path, label_key, roi=roi, n_samples=samples_per_ds[i], ndim=ndim, **kwargs
            )
            dset.rf_paths = rf_paths
            dset.filter_config = filter_config
            dset.rf_channels = rf_channels
            dset.is_anisotropic = is_anisotropic
            ds.append(dset)
        ds = ConcatDataset(*ds)
    return ds


def _load_shallow2deep_image_collection_dataset(
    raw_paths, raw_key, label_paths, label_key, rf_paths, rf_channels, patch_shape, **kwargs
):
    if isinstance(raw_paths, str):
        assert isinstance(label_paths, str)
        raw_file_paths = glob(os.path.join(raw_paths, raw_key))
        raw_file_paths.sort()
        label_file_paths = glob(os.path.join(label_paths, label_key))
        label_file_paths.sort()
        ds = Shallow2DeepImageCollectionDataset(raw_file_paths, label_file_paths, patch_shape, **kwargs)
    elif isinstance(raw_paths, list) and raw_key is None:
        assert isinstance(label_paths, list)
        assert label_key is None
        assert all(os.path.exists(pp) for pp in raw_paths)
        assert all(os.path.exists(pp) for pp in label_paths)
        ds = Shallow2DeepImageCollectionDataset(raw_paths, label_paths, patch_shape, **kwargs)
    else:
        raise NotImplementedError

    filter_config = kwargs.pop("filter_config", None)
    ds.rf_paths = rf_paths
    ds.filter_config = filter_config
    ds.rf_channels = rf_channels
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
    filter_config=None,
    rf_channels=(1,),
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
            raw_paths if isinstance(raw_paths, str) else raw_paths[0], raw_key, is_seg_dataset,
            3 if ndim == "anisotropic" else ndim
        )

    if is_seg_dataset:
        ds = _load_shallow2deep_segmentation_dataset(
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
            filter_config=filter_config,
            rf_channels=rf_channels,
        )
    else:
        if rois is not None:
            raise NotImplementedError
        ds = _load_shallow2deep_image_collection_dataset(
            raw_paths, raw_key, label_paths, label_key, rf_paths, rf_channels, patch_shape,
            raw_transform=raw_transform, label_transform=label_transform,
            transform=transform, dtype=dtype, n_samples=n_samples,
        )
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
    sampler=None,
    ndim=None,
    is_seg_dataset=None,
    with_channels=False,
    rf_channels=(1,),
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
        sampler=sampler,
        ndim=ndim,
        is_seg_dataset=is_seg_dataset,
        with_channels=with_channels,
        filter_config=filter_config,
        rf_channels=rf_channels,
    )
    return get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
