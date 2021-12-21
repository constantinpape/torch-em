import os
import pickle

import numpy as np
import torch
from elf.io import open_file
from torch_em.segmentation import (get_data_loader, get_raw_transform,
                                   samples_to_datasets, _get_default_transform)
from torch_em.data import ConcatDataset, RawDataset
from .prepare_shallow2deep import _get_filters, _apply_filters
from ..util import ensure_tensor_with_channels


class Shallow2DeepDataset(RawDataset):
    _rf_paths = None
    _feature_config = None

    @property
    def rf_paths(self):
        return self._rf_paths

    @rf_paths.setter
    def rf_paths(self, value):
        self._rf_paths = value

    @property
    def feature_config(self):
        return self._feature_config

    @feature_config.setter
    def _feature_config(self, value):
        self._feature_config = value

    def _predict_rf(self, raw):
        n_rfs = len(self._rf_paths)
        rf_path = self._rf_paths[np.random.randint(0, n_rfs)]
        filters_and_sigmas = _get_filters(self._feature_config)
        with open(rf_path, "rb") as f:
            rf = pickle.load(f)
        filters_and_sigmas = _get_filters(self._feature_config)
        features = _apply_filters(raw, filters_and_sigmas)
        prediction = rf.predict_proba(features).reshape(raw.shape)
        return prediction

    def __getitem__(self, index):
        assert self._rf_paths is not None
        assert self._feature_config is not None
        raw = self._get_sample(index)
        prediction = self._predict_rf(raw)
        if self.raw_transform is not None:
            raw = self.raw_transform(raw)
        if self.transform is not None:
            raw = self.transform(raw)
            if self.trafo_halo is not None:
                raw = self.crop(raw)
        raw = ensure_tensor_with_channels(raw, ndim=self._ndim, dtype=self.dtype)
        prediction = ensure_tensor_with_channels(prediction, ndim=self._ndim, dtype=self.dtype)
        return raw, prediction


def check_paths(raw_paths):
    def _check_path(path):
        if not os.path.exists(path):
            raise ValueError(f"Could not find path {path}")

    if isinstance(raw_paths, str):
        _check_path(raw_paths)
    else:
        for rp in zip(raw_paths):
            _check_path(rp)


def _is_raw_dataset(raw_paths, raw_key):
    def _can_open(path, key):
        try:
            open_file(path, mode="r")[key]
            return True
        except Exception:
            return False

    if isinstance(raw_paths, str):
        can_open_raw = _can_open(raw_paths, raw_key)
    else:
        can_open_raw = [_can_open(rp, raw_key) for rp in raw_paths]
        if not can_open_raw.count(can_open_raw[0]) == len(can_open_raw):
            raise ValueError("Inconsistent raw data")
        can_open_raw = can_open_raw[0]
    return can_open_raw


def _load_shallow2deep_dataset(raw_paths, raw_key, rf_paths, **kwargs):
    rois = kwargs.pop("rois", None)
    if isinstance(raw_paths, str):
        if rois is not None:
            assert len(rois) == 3 and all(isinstance(roi, slice) for roi in rois)
        ds = Shallow2DeepDataset(raw_paths, raw_key, roi=rois, **kwargs)
        ds.rf_paths = rf_paths
    else:
        assert len(raw_paths) > 0
        if rois is not None:
            assert all(isinstance(roi, tuple) for roi in rois)
        n_samples = kwargs.pop("n_samples", None)
        samples_per_ds = (
            [None] * len(raw_paths) if n_samples is None else samples_to_datasets(n_samples, raw_paths, raw_key)
        )
        ds = []
        for i, raw_path in enumerate(raw_paths):
            roi = None if rois is None else rois[i]
            dset = Shallow2DeepDataset(
                raw_path, raw_key, roi=roi, n_samples=samples_per_ds[i], **kwargs
            )
            dset.rf_paths = rf_paths
            ds.append(dset)
        ds = ConcatDataset(*ds)
    return ds


def get_shallow2deep_dataset(
    raw_paths,
    raw_key,
    rf_paths,
    patch_shape,
    raw_transform=None,
    transform=None,
    dtype=torch.float32,
    rois=None,
    n_samples=None,
    sampler=None,
    ndim=None,
    is_raw_dataset=None,
    with_channels=False,
):
    check_paths(raw_paths)
    if is_raw_dataset is None:
        is_raw_dataset = _is_raw_dataset(raw_paths, raw_key)

    # we always use a raw transform in the convenience function
    if raw_transform is None:
        raw_transform = get_raw_transform()

    # we always use augmentations in the convenience function
    if transform is None:
        transform = _get_default_transform(
            raw_paths if isinstance(raw_paths, str) else raw_paths[0], raw_key, is_raw_dataset, ndim
        )

    if is_raw_dataset:
        ds = _load_shallow2deep_dataset(
            raw_paths,
            raw_key,
            rf_paths,
            patch_shape=patch_shape,
            raw_transform=raw_transform,
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
    rf_paths,
    batch_size,
    patch_shape,
    feature_config=None,
    raw_transform=None,
    transform=None,
    rois=None,
    n_samples=None,
    ndim=None,
    is_raw_dataset=None,
    with_channels=False,
    **loader_kwargs,
):
    ds = get_shallow2deep_dataset(
        raw_paths=raw_paths,
        raw_key=raw_key,
        rf_paths=rf_paths,
        patch_shape=patch_shape,
        raw_transform=raw_transform,
        transform=transform,
        rois=rois,
        n_samples=n_samples,
        ndim=ndim,
        is_raw_dataset=is_raw_dataset,
        with_channels=with_channels,
    )
    return get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
