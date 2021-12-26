import os
import pickle
from concurrent import futures
from glob import glob
from functools import partial

import numpy as np
import torch_em
from sklearn.ensemble import RandomForestClassifier
from torch_em.segmentation import check_paths, is_segmentation_dataset, samples_to_datasets
from tqdm import tqdm

try:
    import fastfilters as filter_impl
except ImportError:
    import vigra.filters as filter_impl


class RFSegmentationDataset(torch_em.data.SegmentationDataset):
    _patch_shape_min = None
    _patch_shape_max = None

    @property
    def patch_shape_min(self):
        return self._patch_shape_min

    @patch_shape_min.setter
    def patch_shape_min(self, value):
        self._patch_shape_min = value

    @property
    def patch_shape_max(self):
        return self._patch_shape_max

    @patch_shape_max.setter
    def patch_shape_max(self, value):
        self._patch_shape_max = value

    def _sample_bounding_box(self):
        assert self._patch_shape_min is not None and self._patch_shape_max is not None
        sample_shape = [
            pmin if pmin == pmax else np.random.randint(pmin, pmax)
            for pmin, pmax in zip(self._patch_shape_min, self._patch_shape_max)
        ]
        bb_start = [
            np.random.randint(0, sh - psh) if sh - psh > 0 else 0
            for sh, psh in zip(self.shape, sample_shape)
        ]
        return tuple(slice(start, start + psh) for start, psh in zip(bb_start, sample_shape))


class RFImageCollectionDataset(torch_em.data.ImageCollectionDataset):
    _patch_shape_min = None
    _patch_shape_max = None

    @property
    def patch_shape_min(self):
        return self._patch_shape_min

    @patch_shape_min.setter
    def patch_shape_min(self, value):
        self._patch_shape_min = value

    @property
    def patch_shape_max(self):
        return self._patch_shape_max

    @patch_shape_max.setter
    def patch_shape_max(self, value):
        self._patch_shape_max = value

    def _sample_bounding_box(self, shape):
        if any(sh < psh for sh, psh in zip(shape, self.patch_shape_max)):
            raise NotImplementedError("Image padding is not supported yet.")
        assert self._patch_shape_min is not None and self._patch_shape_max is not None
        patch_shape = [
            pmin if pmin == pmax else np.random.randint(pmin, pmax)
            for pmin, pmax in zip(self._patch_shape_min, self._patch_shape_max)
        ]
        bb_start = [
            np.random.randint(0, sh - psh) if sh - psh > 0 else 0 for sh, psh in zip(shape, patch_shape)
        ]
        return tuple(slice(start, start + psh) for start, psh in zip(bb_start, patch_shape))


def _load_rf_segmentation_dataset(
    raw_paths, raw_key, label_paths, label_key, patch_shape_min, patch_shape_max, **kwargs
):
    rois = kwargs.pop("rois", None)
    if isinstance(raw_paths, str):
        if rois is not None:
            assert len(rois) == 3 and all(isinstance(roi, slice) for roi in rois)
        ds = RFSegmentationDataset(
            raw_paths, raw_key, label_paths, label_key, roi=rois, patch_shape=patch_shape_min, **kwargs
        )
        ds.patch_shape_min = patch_shape_min
        ds.patch_shape_max = patch_shape_max
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
            dset = RFSegmentationDataset(
                raw_path, raw_key, label_path, label_key, roi=roi, n_samples=samples_per_ds[i],
                patch_shape=patch_shape_min, **kwargs
            )
            dset.patch_shape_min = patch_shape_min
            dset.patch_shape_max = patch_shape_max
            ds.append(dset)
        ds = torch_em.data.ConcatDataset(*ds)
    return ds


def _load_rf_image_collection_dataset(
    raw_paths, raw_key, label_paths, label_key, patch_shape_min, patch_shape_max, roi, **kwargs
):
    def _get_paths(rpath, rkey, lpath, lkey, this_roi):
        rpath = glob(os.path.join(rpath, rkey))
        rpath.sort()
        if len(rpath) == 0:
            raise ValueError(f"Could not find any images for pattern {os.path.join(rpath, rkey)}")
        lpath = glob(os.path.join(lpath, lkey))
        lpath.sort()
        if len(rpath) != len(lpath):
            raise ValueError(f"Expect same number of raw and label images, got {len(rpath)}, {len(lpath)}")

        if this_roi is not None:
            rpath, lpath = rpath[roi], lpath[roi]

        return rpath, lpath

    def _check_patch(patch_shape):
        if len(patch_shape) == 3:
            if patch_shape[0] != 1:
                raise ValueError(f"Image collection dataset expects 2d patch shape, got {patch_shape}")
            patch_shape = patch_shape[1:]
        assert len(patch_shape) == 2
        return patch_shape

    patch_shape_min = _check_patch(patch_shape_min)
    patch_shape_max = _check_patch(patch_shape_max)

    if isinstance(raw_paths, str):
        raw_paths, label_paths = _get_paths(raw_paths, raw_key, label_paths, label_key, roi)
        ds = RFImageCollectionDataset(raw_paths, label_paths, patch_shape=patch_shape_min, **kwargs)
    elif raw_key is None:
        assert label_key is None
        assert isinstance(raw_paths, (list, tuple)) and isinstance(label_paths, (list, tuple))
        assert len(raw_paths) == len(label_paths)
        ds = RFImageCollectionDataset(raw_paths, label_paths, patch_shape=patch_shape_min, **kwargs)
        ds.patch_shape_min = patch_shape_min
        ds.patch_shape_max = patch_shape_max
    else:
        ds = []
        n_samples = kwargs.pop("n_samples", None)
        samples_per_ds = (
            [None] * len(raw_paths) if n_samples is None else samples_to_datasets(n_samples, raw_paths, raw_key)
        )
        if roi is None:
            roi = len(raw_paths) * [None]
        assert len(roi) == len(raw_paths)
        for i, (raw_path, label_path, this_roi) in enumerate(zip(raw_paths, label_paths, roi)):
            rpath, lpath = _get_paths(raw_path, raw_key, label_path, label_key, this_roi)
            dset = RFImageCollectionDataset(
                rpath, lpath, patch_shape=patch_shape_min, n_samples=samples_per_ds[i], **kwargs
            )
            dset.patch_shape_min = patch_shape_min
            dset.patch_shape_max = patch_shape_max
            ds.append(dset)
        ds = torch_em.data.ConcatDataset(*ds)
    return ds


def _get_filters(ndim, filters_and_sigmas):
    # subset of ilastik default features
    if filters_and_sigmas is None:
        filters = [filter_impl.gaussianSmoothing,
                   filter_impl.laplacianOfGaussian,
                   filter_impl.gaussianGradientMagnitude,
                   filter_impl.hessianOfGaussianEigenvalues,
                   filter_impl.structureTensorEigenvalues]
        sigmas = [0.7, 1.6, 3.5, 5.0]
        filters_and_sigmas = [
            (filt, sigma) if i != len(filters) - 1 else (partial(filt, outerScale=2*sigma), sigma)
            for i, filt in enumerate(filters) for sigma in sigmas
        ]
    # validate the filter config
    assert isinstance(filters_and_sigmas, (list, tuple))
    for filt_and_sig in filters_and_sigmas:
        filt, sig = filt_and_sig
        assert callable(filt) or (isinstance(filt, str) and hasattr(filter_impl, filt))
        assert isinstance(sig, (float, tuple))
        if isinstance(sig, tuple):
            assert ndim is not None and len(sig) == ndim
            assert all(isinstance(sigg, float) for sigg in sig)
    return filters_and_sigmas


def _apply_filters(raw, filters_and_sigmas):
    features = []
    for filter_, sigma in filters_and_sigmas:
        func = getattr(filter_impl, filter_) if isinstance(filter_, str) else filter_
        response = func(raw, sigma)
        if response.ndim > raw.ndim:
            for c in range(response.shape[-1]):
                features.append(response[..., c].flatten())
        else:
            features.append(response.flatten())
    features = np.concatenate([ff[:, None] for ff in features], axis=1)
    return features


def _apply_filters_with_mask(raw, filters_and_sigmas, mask):
    features = []
    for filter_, sigma in filters_and_sigmas:
        func = getattr(filter_impl, filter_) if isinstance(filter_, str) else filter_
        response = func(raw, sigma)
        if response.ndim > raw.ndim:
            for c in range(response.shape[-1]):
                features.append(response[..., c][mask])
        else:
            features.append(response[mask])
    features = np.concatenate([ff[:, None] for ff in features], axis=1)
    return features


def _get_features_and_labels(raw, labels, filters_and_sigmas, balance_labels):
    # find the mask for where we compute filters and labels
    # by default we exclude everything that has label -1
    assert labels.shape == raw.shape
    mask = labels != -1
    if balance_labels:
        label_ids, label_counts = np.unique(labels[mask], return_counts=True)
        if not np.array_equal(label_ids, np.array([0, 1])):
            raise NotImplementedError(f"Label balancing is only implemented for binary lables, got {label_ids}")
        n_labels = label_counts[np.argmin(label_counts)]
        n_discard = label_counts[np.argmax(label_counts)] - n_labels
        max_label = label_ids[np.argmax(label_counts)]
        # sample from the maximum label
        # shuffle the positions and only keep up to n_labels in the mask
        max_label_pos = np.where(labels == max_label)
        discard_ids = np.arange(len(max_label_pos[0]))
        np.random.shuffle(discard_ids)
        discard_ids = discard_ids[:n_discard]
        discard_mask = tuple(pos[discard_ids] for pos in max_label_pos)
        mask[discard_mask] = False
        assert mask.sum() == 2 * n_labels
    labels = labels[mask]
    assert labels.ndim == 1
    features = _apply_filters_with_mask(raw, filters_and_sigmas, mask)
    assert features.ndim == 2
    assert len(features) == len(labels)
    return features, labels


def prepare_shallow2deep(
    raw_paths,
    raw_key,
    label_paths,
    label_key,
    patch_shape_min,
    patch_shape_max,
    n_forests,
    n_threads,
    output_folder,
    ndim,
    raw_transform=None,
    label_transform=None,
    rois=None,
    is_seg_dataset=None,
    balance_labels=True,
    filter_config=None,
    **rf_kwargs,
):
    assert len(patch_shape_min) == len(patch_shape_max)
    assert all(maxs >= mins for maxs, mins in zip(patch_shape_max, patch_shape_min))
    check_paths(raw_paths, label_paths)

    # get the correct dataset
    if is_seg_dataset is None:
        is_seg_dataset = is_segmentation_dataset(raw_paths, raw_key, label_paths, label_key)
    if is_seg_dataset:
        ds = _load_rf_segmentation_dataset(raw_paths, raw_key, label_paths, label_key,
                                           patch_shape_min, patch_shape_max,
                                           raw_transform=raw_transform, label_transform=label_transform,
                                           rois=rois, n_samples=n_forests)
    else:
        ds = _load_rf_image_collection_dataset(raw_paths, raw_key, label_paths, label_key,
                                               patch_shape_min, patch_shape_max, roi=rois,
                                               raw_transform=raw_transform, label_transform=label_transform,
                                               n_samples=n_forests)
    assert len(ds) == n_forests, f"{len(ds), {n_forests}}"

    filters_and_sigmas = _get_filters(ndim, filter_config)
    os.makedirs(output_folder, exist_ok=True)

    def _train_rf(rf_id):
        # sample random patch with dataset
        raw, labels = ds[rf_id]
        # cast to numpy and remove channel axis
        # need to update this to support multi-channel input data and/or multi class prediction
        raw, labels = raw.numpy().squeeze(), labels.numpy().astype("int8").squeeze()
        assert raw.ndim == labels.ndim == ndim, f"{raw.ndim}, {labels.ndim}, {ndim}"
        features, labels = _get_features_and_labels(raw, labels, filters_and_sigmas, balance_labels)
        rf = RandomForestClassifier(**rf_kwargs)
        rf.fit(features, labels)
        out_path = os.path.join(output_folder, f"rf_{rf_id}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(rf, f)

    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(tp.map(_train_rf, range(n_forests)), desc="Train RFs", total=n_forests))
