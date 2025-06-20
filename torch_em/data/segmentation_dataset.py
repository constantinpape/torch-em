import os
import warnings
from typing import List, Union, Tuple, Optional, Any, Callable

import numpy as np
from math import ceil

import torch

from elf.wrapper import RoiWrapper

from ..util import ensure_spatial_array, ensure_tensor_with_channels, load_data, ensure_patch_shape


class SegmentationDataset(torch.utils.data.Dataset):
    """Dataset that provides raw data and labels stored in a container data format for segmentation training.

    The dataset loads a patch from the raw and label data and returns a sample for a batch.
    Image data and label data must have the same shape, except for potential channels.
    The dataset supports all file formats that can be opened with `elf.io.open_file`, such as hdf5, zarr or n5.
    Use `raw_path` / `label_path` to specify the file path and `raw_key` / `label_key` to specify the internal dataset.
    It also supports regular image formats, such as .tif. For these cases set `raw_key=None` / `label_key=None`.

    Args:
        raw_path: The file path to the raw image data. May also be a list of file paths.
        raw_key: The key to the internal dataset containing the raw data.
        label_path: The file path to the label data. May also be a list of file paths.
        label_key: The key to the internal dataset containing the label data
        patch_shape: The patch shape for a training sample.
        raw_transform: Transformation applied to the raw data of a sample.
        label_transform: Transformation applied to the label data of a sample,
            before applying augmentations via `transform`.
        label_transform2: Transformation applied to the label data of a sample,
            after applying augmentations via `transform`.
        transform: Transformation applied to both the raw data and label data of a sample.
            This can be used to implement data augmentations.
        roi: Region of interest in the data. If given, the data will only be loaded from the corresponding area.
        dtype: The return data type of the raw data.
        label_dtype: The return data type of the label data.
        n_samples: The length of this dataset. If None, the length will be set to `len(raw_image_paths)`.
        sampler: Sampler for rejecting samples according to a defined criterion.
            The sampler must be a callable that accepts the raw data (as numpy arrays) as input.
        ndim: The spatial dimensionality of the data. If None, will be derived from the raw data.
        with_channels: Whether the raw data has channels.
        with_label_channels: Whether the label data has channels.
        with_padding: Whether to pad samples to `patch_shape` if their shape is smaller.
        z_ext: Extra bounding box for loading the data across z.
    """
    max_sampling_attempts = 500
    """The maximal number of sampling attempts, for loading a sample via `__getitem__`.
    This is used when `sampler` rejects a sample, to avoid an infinite loop if no valid sample can be found.
    """

    @staticmethod
    def compute_len(shape, patch_shape):
        if patch_shape is None:
            return 1
        else:
            n_samples = ceil(np.prod([float(sh / csh) for sh, csh in zip(shape, patch_shape)]))
            return n_samples

    def __init__(
        self,
        raw_path: Union[List[Any], str, os.PathLike],
        raw_key: Optional[str],
        label_path: Union[List[Any], str, os.PathLike],
        label_key: Optional[str],
        patch_shape: Tuple[int, ...],
        raw_transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
        label_transform2: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        roi: Optional[Union[slice, Tuple[slice, ...]]] = None,
        dtype: torch.dtype = torch.float32,
        label_dtype: torch.dtype = torch.float32,
        n_samples: Optional[int] = None,
        sampler: Optional[Callable] = None,
        ndim: Optional[int] = None,
        with_channels: bool = False,
        with_label_channels: bool = False,
        with_padding: bool = True,
        z_ext: Optional[int] = None,
    ):
        self.raw_path = raw_path
        self.raw_key = raw_key
        self.raw = load_data(raw_path, raw_key)

        self.label_path = label_path
        self.label_key = label_key
        self.labels = load_data(label_path, label_key)

        self._with_channels = with_channels
        self._with_label_channels = with_label_channels

        if roi is not None:
            if isinstance(roi, slice):
                roi = (roi,)

            self.raw = RoiWrapper(self.raw, (slice(None),) + roi) if self._with_channels else RoiWrapper(self.raw, roi)
            self.labels = RoiWrapper(self.labels, (slice(None),) + roi) if self._with_label_channels else\
                RoiWrapper(self.labels, roi)

        shape_raw = self.raw.shape[1:] if self._with_channels else self.raw.shape
        shape_label = self.labels.shape[1:] if self._with_label_channels else self.labels.shape
        assert shape_raw == shape_label, f"{shape_raw}, {shape_label}"

        self.shape = shape_raw
        self.roi = roi

        self._ndim = len(shape_raw) if ndim is None else ndim
        assert self._ndim in (2, 3, 4), f"Invalid data dimensions: {self._ndim}. Only 2d, 3d or 4d data is supported"

        if patch_shape is not None:
            assert len(patch_shape) in (self._ndim, self._ndim + 1), f"{patch_shape}, {self._ndim}"

        self.patch_shape = patch_shape

        self.raw_transform = raw_transform
        self.label_transform = label_transform
        self.label_transform2 = label_transform2
        self.transform = transform
        self.sampler = sampler
        self.with_padding = with_padding

        self.dtype = dtype
        self.label_dtype = label_dtype

        self._len = self.compute_len(self.shape, self.patch_shape) if n_samples is None else n_samples

        self.z_ext = z_ext

        self.sample_shape = patch_shape
        self.trafo_halo = None
        # TODO add support for trafo halo: asking for a bigger bounding box before applying the trafo,
        # which is then cut. See code below; but this ne needs to be properly tested

        # self.trafo_halo = None if self.transform is None else self.transform.halo(self.patch_shape)
        # if self.trafo_halo is not None:
        #     if len(self.trafo_halo) == 2 and self._ndim == 3:
        #         self.trafo_halo = (0,) + self.trafo_halo
        #     assert len(self.trafo_halo) == self._ndim
        #     self.sample_shape = tuple(sh + ha for sh, ha in zip(self.patch_shape, self.trafo_halo))
        #     self.inner_bb = tuple(slice(ha, sh - ha) for sh, ha in zip(self.patch_shape, self.trafo_halo))

    def __len__(self):
        return self._len

    @property
    def ndim(self):
        return self._ndim

    def _sample_bounding_box(self):
        if self.sample_shape is None:
            if self.z_ext is None:
                bb_start = [0] * len(self.shape)
                patch_shape_for_bb = self.shape
            else:
                z_diff = self.shape[0] - self.z_ext
                bb_start = [np.random.randint(0, z_diff) if z_diff > 0 else 0] + [0] * len(self.shape[1:])
                patch_shape_for_bb = (self.z_ext, *self.shape[1:])

        else:
            bb_start = [
                np.random.randint(0, sh - psh) if sh - psh > 0 else 0 for sh, psh in zip(self.shape, self.sample_shape)
            ]
            patch_shape_for_bb = self.sample_shape

        return tuple(slice(start, start + psh) for start, psh in zip(bb_start, patch_shape_for_bb))

    def _get_desired_raw_and_labels(self):
        bb = self._sample_bounding_box()
        bb_raw = (slice(None),) + bb if self._with_channels else bb
        bb_labels = (slice(None),) + bb if self._with_label_channels else bb
        raw, labels = self.raw[bb_raw], self.labels[bb_labels]
        return raw, labels

    def _get_sample(self, index):
        if self.raw is None or self.labels is None:
            raise RuntimeError("SegmentationDataset has not been properly deserialized.")

        raw, labels = self._get_desired_raw_and_labels()

        if self.sampler is not None:
            sample_id = 0
            while not self.sampler(raw, labels):
                raw, labels = self._get_desired_raw_and_labels()
                sample_id += 1
                if sample_id > self.max_sampling_attempts:
                    raise RuntimeError(f"Could not sample a valid batch in {self.max_sampling_attempts} attempts")

        # Padding the patch to match the expected input shape.
        if self.patch_shape is not None and self.with_padding:
            raw, labels = ensure_patch_shape(
                raw=raw,
                labels=labels,
                patch_shape=self.patch_shape,
                have_raw_channels=self._with_channels,
                have_label_channels=self._with_label_channels,
            )

        # squeeze the singleton spatial axis if we have a spatial shape that is larger by one than self._ndim
        if self.patch_shape is not None and len(self.patch_shape) == self._ndim + 1:
            raw = raw.squeeze(1 if self._with_channels else 0)
            labels = labels.squeeze(1 if self._with_label_channels else 0)

        return raw, labels

    def crop(self, tensor):
        """@private
        """
        bb = self.inner_bb
        if tensor.ndim > len(bb):
            bb = (tensor.ndim - len(bb)) * (slice(None),) + bb
        return tensor[bb]

    def __getitem__(self, index):
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

        raw = ensure_tensor_with_channels(raw, ndim=self._ndim, dtype=self.dtype)
        labels = ensure_tensor_with_channels(labels, ndim=self._ndim, dtype=self.label_dtype)
        return raw, labels

    # need to overwrite pickle to support h5py
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["raw"]
        del state["labels"]
        return state

    def __setstate__(self, state):
        raw_path, raw_key = state["raw_path"], state["raw_key"]
        label_path, label_key = state["label_path"], state["label_key"]
        roi = state["roi"]
        try:
            raw = load_data(raw_path, raw_key)
            if roi is not None:
                raw = RoiWrapper(raw, (slice(None),) + roi) if state["_with_channels"] else RoiWrapper(raw, roi)
            state["raw"] = raw
        except Exception:
            msg = f"SegmentationDataset could not be deserialized because of missing {raw_path}, {raw_key}.\n"
            msg += "The dataset is deserialized in order to allow loading trained models from a checkpoint.\n"
            msg += "But it cannot be used for further training and will throw an error."
            warnings.warn(msg)
            state["raw"] = None

        try:
            labels = load_data(label_path, label_key)
            if roi is not None:
                labels = RoiWrapper(labels, (slice(None),) + roi) if state["_with_label_channels"] else\
                    RoiWrapper(labels, roi)
            state["labels"] = labels
        except Exception:
            msg = f"SegmentationDataset could not be deserialized because of missing {label_path}, {label_key}.\n"
            msg += "The dataset is deserialized in order to allow loading trained models from a checkpoint.\n"
            msg += "But it cannot be used for further training and will throw an error."
            warnings.warn(msg)
            state["labels"] = None

        self.__dict__.update(state)
