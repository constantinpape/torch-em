import os
import copy
import pickle
import warnings
from concurrent import futures
from glob import glob
from functools import partial
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch_em
from scipy.ndimage import gaussian_filter, convolve
from skimage.feature import peak_local_max
from sklearn.ensemble import RandomForestClassifier
from torch_em.segmentation import check_paths, is_segmentation_dataset, samples_to_datasets
from tqdm import tqdm

import vigra
try:
    import fastfilters as filter_impl
except ImportError:
    import vigra.filters as filter_impl


class RFSegmentationDataset(torch_em.data.SegmentationDataset):
    """@private
    """
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
    """@private
    """
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
    sampler = kwargs.pop("sampler", None)
    sampler = sampler if sampler else torch_em.data.MinForegroundSampler(min_fraction=0.01)
    if isinstance(raw_paths, str):
        if rois is not None:
            assert len(rois) == 3 and all(isinstance(roi, slice) for roi in rois)
        ds = RFSegmentationDataset(
            raw_paths, raw_key, label_paths, label_key, roi=rois, patch_shape=patch_shape_min, sampler=sampler, **kwargs
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
                patch_shape=patch_shape_min, sampler=sampler, **kwargs
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
        ds.patch_shape_min = patch_shape_min
        ds.patch_shape_max = patch_shape_max
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
            (filt, sigma) if i != len(filters) - 1 else (partial(filt, outerScale=0.5*sigma), sigma)
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


def _calculate_response(raw, filter_, sigma):
    if callable(filter_):
        return filter_(raw, sigma)

    # filter_ is still string, convert it to function
    # fastfilters does not support passing sigma as tuple
    func = getattr(vigra.filters, filter_) if isinstance(sigma, tuple) else getattr(filter_impl, filter_)

    # special case since additional argument outerScale
    # is needed for structureTensorEigenvalues functions
    if filter_ == "structureTensorEigenvalues":
        outerScale = tuple([s*2 for s in sigma]) if isinstance(sigma, tuple) else 2*sigma
        return func(raw, sigma, outerScale=outerScale)

    return func(raw, sigma)


def _apply_filters(raw, filters_and_sigmas):
    features = []
    for filter_, sigma in filters_and_sigmas:
        response = _calculate_response(raw, filter_, sigma)
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
        response = _calculate_response(raw, filter_, sigma)
        if response.ndim > raw.ndim:
            for c in range(response.shape[-1]):
                features.append(response[..., c][mask])
        else:
            features.append(response[mask])
    features = np.concatenate([ff[:, None] for ff in features], axis=1)
    return features


def _balance_labels(labels, mask):
    class_ids, label_counts = np.unique(labels[mask], return_counts=True)
    n_classes = len(class_ids)
    assert class_ids.tolist() == list(range(n_classes))

    min_class = class_ids[np.argmin(label_counts)]
    n_labels = label_counts[min_class]

    for class_id in class_ids:
        if class_id == min_class:
            continue
        n_discard = label_counts[class_id] - n_labels
        # sample from the current class
        # shuffle the positions and only keep up to n_labels in the mask
        label_pos = np.where(labels == class_id)
        discard_ids = np.arange(len(label_pos[0]))
        np.random.shuffle(discard_ids)
        discard_ids = discard_ids[:n_discard]
        discard_mask = tuple(pos[discard_ids] for pos in label_pos)
        mask[discard_mask] = False

    assert mask.sum() == n_classes * n_labels
    return mask


def _get_features_and_labels(raw, labels, filters_and_sigmas, balance_labels, return_mask=False):
    # find the mask for where we compute filters and labels
    # by default we exclude everything that has label -1
    assert labels.shape == raw.shape
    mask = labels != -1
    if balance_labels:
        mask = _balance_labels(labels, mask)
    labels = labels[mask]
    assert labels.ndim == 1
    features = _apply_filters_with_mask(raw, filters_and_sigmas, mask)
    assert features.ndim == 2
    assert len(features) == len(labels)
    if return_mask:
        return features, labels, mask
    else:
        return features, labels


def _prepare_shallow2deep(
    raw_paths,
    raw_key,
    label_paths,
    label_key,
    patch_shape_min,
    patch_shape_max,
    n_forests,
    ndim,
    raw_transform,
    label_transform,
    rois,
    is_seg_dataset,
    filter_config,
    sampler,
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
                                           rois=rois, n_samples=n_forests, sampler=sampler)
    else:
        ds = _load_rf_image_collection_dataset(raw_paths, raw_key, label_paths, label_key,
                                               patch_shape_min, patch_shape_max, roi=rois,
                                               raw_transform=raw_transform, label_transform=label_transform,
                                               n_samples=n_forests)

    assert len(ds) == n_forests, f"{len(ds), {n_forests}}"
    filters_and_sigmas = _get_filters(ndim, filter_config)
    return ds, filters_and_sigmas


def _serialize_feature_config(filters_and_sigmas):
    feature_config = [
        (filt if isinstance(filt, str) else (filt.func.__name__ if isinstance(filt, partial) else filt.__name__), sigma)
        for filt, sigma in filters_and_sigmas
    ]
    return feature_config


def prepare_shallow2deep(
    raw_paths: Union[str, Sequence[str]],
    raw_key: Optional[str],
    label_paths: Union[str, Sequence[str]],
    label_key: Optional[str],
    patch_shape_min: Tuple[int, ...],
    patch_shape_max: Tuple[int, ...],
    n_forests: int,
    n_threads: int,
    output_folder: str,
    ndim: int,
    raw_transform: Optional[Callable] = None,
    label_transform: Optional[Callable] = None,
    rois: Optional[Union[Tuple[slice, ...], Sequence[Tuple[slice, ...]]]] = None,
    is_seg_dataset: Optional[bool] = None,
    balance_labels: bool = True,
    filter_config: Optional[Dict] = None,
    sampler: Optional[Callable] = None,
    **rf_kwargs,
) -> None:
    """Prepare shallow2deep enhancer training by pre-training random forests.

    Args:
        raw_paths: The file paths to the raw data. May also be a single file.
        raw_key: The name of the internal dataset for the raw data. Set to None for regular image such as tif.
        label_paths: The file paths to the lable data. May also be a single file.
        label_key: The name of the internal dataset for the label data. Set to None for regular image such as tif.
        patch_shape_min: The minimal patch shape loaded for training a random forest.
        patch_shape_max: The maximal patch shape loaded for training a random forest.
        n_forests: The number of random forests to train.
        n_threads: The number of threads for parallelizing the training.
        output_folder: The folder for saving the random forests.
        ndim: The dimensionality of the data.
        raw_transform: A transform to apply to the raw data before computing feautres on it.
        label_transform: A transform to apply to the label data before deriving targets for the random forest for it.
        rois: Region of interests for the training data.
        is_seg_dataset: Whether to create a segmentation dataset or an image collection dataset.
            If None, this wil be determined from the data.
        balance_labels: Whether to balance the training labels for the random forest.
        filter_config: The configuration for the image filters that are used to compute features for the random forest.
        sampler: A sampler to reject samples from training.
        rf_kwargs: Keyword arguments for creating the random forest.
    """
    os.makedirs(output_folder, exist_ok=True)
    ds, filters_and_sigmas = _prepare_shallow2deep(
        raw_paths, raw_key, label_paths, label_key,
        patch_shape_min, patch_shape_max, n_forests, ndim,
        raw_transform, label_transform, rois, is_seg_dataset,
        filter_config, sampler,
    )
    serialized_feature_config = _serialize_feature_config(filters_and_sigmas)

    def _train_rf(rf_id):
        # Sample random patch with dataset.
        raw, labels = ds[rf_id]
        # Cast to numpy and remove channel axis.
        # Need to update this to support multi-channel input data and/or multi class prediction.
        raw, labels = raw.numpy().squeeze(), labels.numpy().astype("int8").squeeze()
        assert raw.ndim == labels.ndim == ndim, f"{raw.ndim}, {labels.ndim}, {ndim}"
        features, labels = _get_features_and_labels(raw, labels, filters_and_sigmas, balance_labels)
        rf = RandomForestClassifier(**rf_kwargs)
        rf.fit(features, labels)
        # Monkey patch these so that we know the feature config and dimensionality.
        rf.feature_ndim = ndim
        rf.feature_config = serialized_feature_config
        out_path = os.path.join(output_folder, f"rf_{rf_id:04d}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(rf, f)

    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(tp.map(_train_rf, range(n_forests)), desc="Train RFs", total=n_forests))


def _score_based_points(
    score_function,
    features, labels, rf_id,
    forests, forests_per_stage,
    sample_fraction_per_stage,
    accumulate_samples,
):
    # get the corresponding random forest from the last stage
    # and predict with it
    last_forest = forests[rf_id - forests_per_stage]
    pred = last_forest.predict_proba(features)

    score = score_function(pred, labels)
    assert len(score) == len(features)

    # get training samples based on the label-prediction diff
    samples = []
    nc = len(np.unique(labels))
    # sample in a class balanced way
    n_samples = int(sample_fraction_per_stage * len(features))
    n_samples_class = n_samples // nc
    for class_id in range(nc):
        class_indices = np.where(labels == class_id)[0]
        this_samples = class_indices[np.argsort(score[class_indices])[::-1][:n_samples_class]]
        samples.append(this_samples)
    samples = np.concatenate(samples)

    # get the features and labels, add from previous rf if specified
    features, labels = features[samples], labels[samples]
    if accumulate_samples:
        features = np.concatenate([last_forest.train_features, features], axis=0)
        labels = np.concatenate([last_forest.train_labels, labels], axis=0)

    return features, labels


def worst_points(
    features, labels, rf_id,
    forests, forests_per_stage,
    sample_fraction_per_stage,
    accumulate_samples=True,
    **kwargs,
):
    """@private
    """
    def score(pred, labels):
        # labels to one-hot encoding
        unique, inverse = np.unique(labels, return_inverse=True)
        onehot = np.eye(unique.shape[0])[inverse]
        # compute the difference between labels and prediction
        return np.abs(onehot - pred).sum(axis=1)

    return _score_based_points(
        score, features, labels, rf_id, forests, forests_per_stage, sample_fraction_per_stage, accumulate_samples
    )


def uncertain_points(
    features, labels, rf_id,
    forests, forests_per_stage,
    sample_fraction_per_stage,
    accumulate_samples=True,
    **kwargs,
):
    """@private
    """
    def score(pred, labels):
        assert pred.ndim == 2
        channel_sorted = np.sort(pred, axis=1)
        uncertainty = channel_sorted[:, -1] - channel_sorted[:, -2]
        return uncertainty

    return _score_based_points(
        score, features, labels, rf_id, forests, forests_per_stage, sample_fraction_per_stage, accumulate_samples
    )


def uncertain_worst_points(
    features, labels, rf_id,
    forests, forests_per_stage,
    sample_fraction_per_stage,
    accumulate_samples=True,
    alpha=0.5,
    **kwargs,
):
    """@private
    """
    def score(pred, labels):
        assert pred.ndim == 2

        # labels to one-hot encoding
        unique, inverse = np.unique(labels, return_inverse=True)
        onehot = np.eye(unique.shape[0])[inverse]
        # compute the difference between labels and prediction
        diff = np.abs(onehot - pred).sum(axis=1)

        channel_sorted = np.sort(pred, axis=1)
        uncertainty = channel_sorted[:, -1] - channel_sorted[:, -2]
        return alpha * diff + (1.0 - alpha) * uncertainty

    return _score_based_points(
        score, features, labels, rf_id, forests, forests_per_stage, sample_fraction_per_stage, accumulate_samples
    )


def random_points(
    features, labels, rf_id,
    forests, forests_per_stage,
    sample_fraction_per_stage,
    accumulate_samples=True,
    **kwargs,
):
    """@private
    """
    samples = []
    nc = len(np.unique(labels))
    # sample in a class balanced way
    n_samples = int(sample_fraction_per_stage * len(features))
    n_samples_class = n_samples // nc
    for class_id in range(nc):
        class_indices = np.where(labels == class_id)[0]
        this_samples = np.random.choice(
            class_indices, size=n_samples_class, replace=len(class_indices) < n_samples_class
        )
        samples.append(this_samples)
    samples = np.concatenate(samples)
    features, labels = features[samples], labels[samples]

    if accumulate_samples and rf_id >= forests_per_stage:
        last_forest = forests[rf_id - forests_per_stage]
        features = np.concatenate([last_forest.train_features, features], axis=0)
        labels = np.concatenate([last_forest.train_labels, labels], axis=0)

    return features, labels


def worst_tiles(
    features, labels, rf_id,
    forests, forests_per_stage,
    sample_fraction_per_stage,
    img_shape,
    mask,
    tile_shape=[25, 25],
    smoothing_sigma=None,
    accumulate_samples=True,
    **kwargs,
):
    """@private
    """
    # check inputs
    ndim = len(img_shape)
    assert ndim in [2, 3], img_shape
    assert len(tile_shape) == ndim, tile_shape

    # get the corresponding random forest from the last stage
    # and predict with it
    last_forest = forests[rf_id - forests_per_stage]
    pred = last_forest.predict_proba(features)

    # labels to one-hot encoding
    unique, inverse = np.unique(labels, return_inverse=True)
    onehot = np.eye(unique.shape[0])[inverse]

    # compute the difference between labels and prediction
    diff = np.abs(onehot - pred)
    assert len(diff) == len(features)

    # reshape diff to image shape
    # we need to also take into account the mask here, and if we apply any masking
    # because we can't directly reshape if we have it
    if mask.sum() != mask.size:
        # get the diff image
        diff_img = np.zeros(img_shape + diff.shape[-1:], dtype=diff.dtype)
        diff_img[mask] = diff
        # inflate the features
        full_features = np.zeros((mask.size,) + features.shape[-1:], dtype=features.dtype)
        full_features[mask.ravel()] = features
        features = full_features
        # inflate the labels (with -1 so this will not be sampled)
        full_labels = np.full(mask.size, -1, dtype="int8")
        full_labels[mask.ravel()] = labels
        labels = full_labels
    else:
        diff_img = diff.reshape(img_shape + (-1,))

    # get the number of classes (not counting ignore label)
    class_ids = np.unique(labels)
    nc = len(class_ids) - 1 if -1 in class_ids else len(class_ids)

    # sample in a class balanced way
    n_samples_class = int(sample_fraction_per_stage * len(features)) // nc
    samples = []
    for class_id in range(nc):
        # smooth either with gaussian or 1-kernel
        if smoothing_sigma:
            diff_img_smooth = gaussian_filter(diff_img[..., class_id], smoothing_sigma, mode="constant")
        else:
            kernel = np.ones(tile_shape)
            diff_img_smooth = convolve(diff_img[..., class_id], kernel, mode="constant")

        # get training samples based on tiles around maxima of the label-prediction diff
        # do this in a class-specific way to ensure that each class is sampled
        # get maxima of the label-prediction diff (they seem to be sorted already)
        max_centers = peak_local_max(
            diff_img_smooth,
            min_distance=max(tile_shape),
            exclude_border=tuple([s // 2 for s in tile_shape])
        )

        # get indices of tiles around maxima
        tiles = []
        for center in max_centers:
            tile_slice = tuple(
                slice(
                    center[d]-tile_shape[d]//2,
                    center[d]+tile_shape[d]//2 + 1,
                    None
                ) for d in range(ndim)
            )
            grid = np.mgrid[tile_slice]
            samples_in_tile = grid.reshape(ndim, -1)
            samples_in_tile = np.ravel_multi_index(samples_in_tile, img_shape)
            tiles.append(samples_in_tile)

        # this (very rarely) fails due to empty tile list. Since we usually
        # accumulate the features this doesn't hurt much and we can continue
        try:
            tiles = np.concatenate(tiles)
            # take samples that belong to the current class
            this_samples = tiles[labels[tiles] == class_id][:n_samples_class]
            samples.append(this_samples)
        except ValueError:
            pass

    try:
        samples = np.concatenate(samples)
        features, labels = features[samples], labels[samples]

        # get the features and labels, add from previous rf if specified
        if accumulate_samples:
            features = np.concatenate([last_forest.train_features, features], axis=0)
            labels = np.concatenate([last_forest.train_labels, labels], axis=0)
    except ValueError:
        features, labels = last_forest.train_features, last_forest.train_labels
        warnings.warn(
            f"No features were sampled for forest {rf_id} using features of forest {rf_id - forests_per_stage}"
        )

    return features, labels


def balanced_dense_accumulate(
    features, labels, rf_id,
    forests, forests_per_stage,
    sample_fraction_per_stage,
    accumulate_samples=True,
    **kwargs,
):
    """@private
    """
    samples = []
    nc = len(np.unique(labels))
    # sample in a class balanced way
    # take all pixels from minority class
    # and choose same amount from other classes randomly
    n_samples_class = np.unique(labels, return_counts=True)[1].min()
    for class_id in range(nc):
        class_indices = np.where(labels == class_id)[0]
        this_samples = np.random.choice(
            class_indices, size=n_samples_class, replace=len(class_indices) < n_samples_class
        )
        samples.append(this_samples)
    samples = np.concatenate(samples)
    features, labels = features[samples], labels[samples]

    # accumulate
    if accumulate_samples and rf_id >= forests_per_stage:
        last_forest = forests[rf_id - forests_per_stage]
        features = np.concatenate([last_forest.train_features, features], axis=0)
        labels = np.concatenate([last_forest.train_labels, labels], axis=0)

    return features, labels


SAMPLING_STRATEGIES = {
    "random_points": random_points,
    "uncertain_points": uncertain_points,
    "uncertain_worst_points": uncertain_worst_points,
    "worst_points": worst_points,
    "worst_tiles": worst_tiles,
    "balanced_dense_accumulate": balanced_dense_accumulate,
}
"""@private
"""


def prepare_shallow2deep_advanced(
    raw_paths: Union[str, Sequence[str]],
    raw_key: Optional[str],
    label_paths: Union[str, Sequence[str]],
    label_key: Optional[str],
    patch_shape_min: Tuple[int, ...],
    patch_shape_max: Tuple[int, ...],
    n_forests: int,
    n_threads: int,
    output_folder: str,
    ndim: int,
    forests_per_stage: int,
    sample_fraction_per_stage: float,
    sampling_strategy: Union[str, Callable] = "worst_points",
    sampling_kwargs: Dict = {},
    raw_transform: Optional[Callable] = None,
    label_transform: Optional[Callable] = None,
    rois: Optional[Union[Tuple[slice, ...], Sequence[Tuple[slice, ...]]]] = None,
    is_seg_dataset: Optional[bool] = None,
    balance_labels: bool = True,
    filter_config: Optional[Dict] = None,
    sampler: Optional[Callable] = None,
    **rf_kwargs,
) -> None:
    """Prepare shallow2deep enhancer training by pre-training random forests.

    This function implements an advanced training procedure compared to `prepare_shallow2deep`.
    The 'sampling_strategy' argument determines an advnaced sampling strategies,
    which selects the samples to use for training the random forests.
    The random forest training operates in stages, the parameter 'forests_per_stage' determines how many forests
    are trained in each stage, and 'sample_fraction_per_stage' determines which fraction of samples is used per stage.
    The random forests in stage 0 are trained from random balanced labels.
    For the other stages 'sampling_strategy' determines the strategy; it has to be a function with signature
    '(features, labels, forests, rf_id, forests_per_stage, sample_fraction_per_stage)',
    and return the sampled features and labels. See for example the 'worst_points' function.
    Alternatively, one of the pre-defined strategies can be selected by passing one of the following names:
    - "random_poinst": Select random points.
    - "uncertain_points": Select points with the highest uncertainty.
    - "uncertain_worst_points": Select the points with the highest uncertainty and worst accuracies.
    - "worst_points": Select the points with the worst accuracies.
    - "worst_tiles": Selectt the tiles with the worst accuracies.
    - "balanced_dense_accumulate": Balanced dense accumulation.

    Args:
        raw_paths: The file paths to the raw data. May also be a single file.
        raw_key: The name of the internal dataset for the raw data. Set to None for regular image such as tif.
        label_paths: The file paths to the lable data. May also be a single file.
        label_key: The name of the internal dataset for the label data. Set to None for regular image such as tif.
        patch_shape_min: The minimal patch shape loaded for training a random forest.
        patch_shape_max: The maximal patch shape loaded for training a random forest.
        n_forests: The number of random forests to train.
        n_threads: The number of threads for parallelizing the training.
        output_folder: The folder for saving the random forests.
        ndim: The dimensionality of the data.
        forests_per_stage: The number of forests to train per stage.
        sample_fraction_per_stage: The fraction of samples to use per stage.
        sampling_strategy: The sampling strategy.
        sampling_kwargs: The keyword arguments for the sampling strategy.
        raw_transform: A transform to apply to the raw data before computing feautres on it.
        label_transform: A transform to apply to the label data before deriving targets for the random forest for it.
        rois: Region of interests for the training data.
        is_seg_dataset: Whether to create a segmentation dataset or an image collection dataset.
            If None, this wil be determined from the data.
        balance_labels: Whether to balance the training labels for the random forest.
        filter_config: The configuration for the image filters that are used to compute features for the random forest.
        sampler: A sampler to reject samples from training.
        rf_kwargs: Keyword arguments for creating the random forest.
    """
    os.makedirs(output_folder, exist_ok=True)
    ds, filters_and_sigmas = _prepare_shallow2deep(
        raw_paths, raw_key, label_paths, label_key,
        patch_shape_min, patch_shape_max, n_forests, ndim,
        raw_transform, label_transform, rois, is_seg_dataset,
        filter_config, sampler,
    )
    serialized_feature_config = _serialize_feature_config(filters_and_sigmas)

    forests = []
    n_stages = n_forests // forests_per_stage if n_forests % forests_per_stage == 0 else\
        n_forests // forests_per_stage + 1

    if isinstance(sampling_strategy, str):
        assert sampling_strategy in SAMPLING_STRATEGIES, \
            f"Invalid sampling strategy {sampling_strategy}, only support {list(SAMPLING_STRATEGIES.keys())}"
        sampling_strategy = SAMPLING_STRATEGIES[sampling_strategy]
    assert callable(sampling_strategy)

    with tqdm(total=n_forests) as pbar:

        def _train_rf(rf_id):
            # sample random patch with dataset
            raw, labels = ds[rf_id]

            # cast to numpy and remove channel axis
            # need to update this to support multi-channel input data and/or multi class prediction
            raw, labels = raw.numpy().squeeze(), labels.numpy().astype("int8").squeeze()
            assert raw.ndim == labels.ndim == ndim, f"{raw.ndim}, {labels.ndim}, {ndim}"

            # monkey patch original shape to sampling_kwargs
            # deepcopy needed due to multithreading
            current_kwargs = copy.deepcopy(sampling_kwargs)
            current_kwargs["img_shape"] = raw.shape

            # only balance samples for the first (densely trained) rfs
            features, labels, mask = _get_features_and_labels(
                raw, labels, filters_and_sigmas, balance_labels=False, return_mask=True
            )
            if forests:  # we have forests: apply the sampling strategy
                features, labels = sampling_strategy(
                    features, labels, rf_id,
                    forests, forests_per_stage,
                    sample_fraction_per_stage,
                    mask=mask,
                    **current_kwargs,
                )
            else:  # sample randomly
                features, labels = random_points(
                    features, labels, rf_id,
                    forests, forests_per_stage,
                    sample_fraction_per_stage,
                )

            # fit the random forest
            assert len(features) == len(labels)
            rf = RandomForestClassifier(**rf_kwargs)
            rf.fit(features, labels)
            # monkey patch these so that we know the feature config and dimensionality
            rf.feature_ndim = ndim
            rf.feature_config = serialized_feature_config

            # save the random forest, update pbar, return it
            out_path = os.path.join(output_folder, f"rf_{rf_id:04d}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(rf, f)

            # monkey patch the training data and labels so we can re-use it in later stages
            rf.train_features = features
            rf.train_labels = labels

            pbar.update(1)
            return rf

        for stage in range(n_stages):
            pbar.set_description(f"Train RFs for stage {stage}")
            with futures.ThreadPoolExecutor(n_threads) as tp:
                this_forests = list(tp.map(
                    _train_rf, range(forests_per_stage * stage, forests_per_stage * (stage + 1))
                ))
                forests.extend(this_forests)
