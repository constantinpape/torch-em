import os
import warnings
import numpy as np
from typing import List, Union, Tuple, Optional, Any, Callable

import torch

from elf.wrapper import RoiWrapper

from ..util import ensure_tensor_with_channels, ensure_patch_shape, load_data


class RawDataset(torch.utils.data.Dataset):
    """Dataset that provides raw data stored in a container data format for unsupervised training.

    The dataset loads a patch from the raw data and returns a sample for a batch.
    The dataset supports all file formats that can be opened with `elf.io.open_file`, such as hdf5, zarr or n5.
    Use `raw_path` to specify the path to the file and `raw_key` to specify the internal dataset.
    It also supports regular image formats, such as .tif. For these cases set `raw_key=None`.

    The dataset can also be used for contrastive learning that relies on two different views of the same data.
    You can use the `augmentations` argument for this.

    Args:
        raw_path: The file path to the raw image data. May also be a list of file paths.
        raw_key: The key to the internal dataset containing the raw data.
        patch_shape: The patch shape for a training sample.
        raw_transform: Transformation applied to the raw data of a sample.
        transform: Transformation to the raw data. This can be used to implement data augmentations.
        roi: Region of interest in the raw data.
            If given, the raw data will only be loaded from the corresponding area.
        dtype: The return data type of the raw data.
        n_samples: The length of this dataset. If None, the length will be set to `len(raw_image_paths)`.
        sampler: Sampler for rejecting samples according to a defined criterion.
            The sampler must be a callable that accepts the raw data (as numpy arrays) as input.
        ndim: The spatial dimensionality of the data. If None, will be derived from the raw data.
        with_channels: Whether the raw data has channels.
        augmentations: Augmentations for contrastive learning. If given, these need to be two different callables.
            They will be applied to the sampled raw data to return two independent views of the raw data.
    """
    max_sampling_attempts = 500
    """The maximal number of sampling attempts, for loading a sample via `__getitem__`.
    This is used when `sampler` rejects a sample, to avoid an infinite loop if no valid sample can be found.
    """

    @staticmethod
    def compute_len(shape, patch_shape):
        n_samples = int(np.prod([float(sh / csh) for sh, csh in zip(shape, patch_shape)]))
        return n_samples

    def __init__(
        self,
        raw_path: Union[List[Any], str, os.PathLike],
        raw_key: Optional[str],
        patch_shape: Tuple[int, ...],
        raw_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        roi: Optional[Union[slice, Tuple[slice, ...]]] = None,
        dtype: torch.dtype = torch.float32,
        n_samples: Optional[int] = None,
        sampler: Optional[Callable] = None,
        ndim: Optional[int] = None,
        with_channels: bool = False,
        augmentations: Optional[Tuple[Callable, Callable]] = None,
    ):
        self.raw_path = raw_path
        self.raw_key = raw_key
        self.raw = load_data(raw_path, raw_key)

        self._with_channels = with_channels

        if roi is not None:
            if isinstance(roi, slice):
                roi = (roi,)

            self.raw = RoiWrapper(self.raw, (slice(None),) + roi) if self._with_channels else RoiWrapper(self.raw, roi)

        self.shape = self.raw.shape[1:] if self._with_channels else self.raw.shape
        self.roi = roi

        self._ndim = len(self.shape) if ndim is None else ndim
        assert self._ndim in (2, 3, 4), f"Invalid data dimensions: {self._ndim}. Only 2d, 3d or 4d data is supported"

        assert len(patch_shape) in (self._ndim, self._ndim + 1), f"{patch_shape}, {self._ndim}"
        self.patch_shape = patch_shape

        self.raw_transform = raw_transform
        self.transform = transform
        self.sampler = sampler
        self.dtype = dtype

        if augmentations is not None:
            assert len(augmentations) == 2
        self.augmentations = augmentations

        self._len = self.compute_len(self.shape, self.patch_shape) if n_samples is None else n_samples

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
        bb_start = [
            np.random.randint(0, sh - psh) if sh - psh > 0 else 0
            for sh, psh in zip(self.shape, self.sample_shape)
        ]
        return tuple(slice(start, start + psh) for start, psh in zip(bb_start, self.sample_shape))

    def _get_sample(self, index):
        if self.raw is None:
            raise RuntimeError("RawDataset has not been properly deserialized.")
        bb = self._sample_bounding_box()
        raw = self.raw[(slice(None),) + bb] if self._with_channels else self.raw[bb]

        if self.sampler is not None:
            sample_id = 0
            while not self.sampler(raw):
                bb = self._sample_bounding_box()
                raw = self.raw[(slice(None),) + bb] if self._with_channels else self.raw[bb]
                sample_id += 1
                if sample_id > self.max_sampling_attempts:
                    raise RuntimeError(f"Could not sample a valid batch in {self.max_sampling_attempts} attempts")

        if self.patch_shape is not None:
            raw = ensure_patch_shape(
                raw=raw, labels=None, patch_shape=self.patch_shape, have_raw_channels=self._with_channels
            )

        # squeeze the singleton spatial axis if we have a spatial shape that is larger by one than self._ndim
        if len(self.patch_shape) == self._ndim + 1:
            raw = raw.squeeze(1 if self._with_channels else 0)

        return raw

    def crop(self, tensor):
        bb = self.inner_bb
        if tensor.ndim > len(bb):
            bb = (tensor.ndim - len(bb)) * (slice(None),) + bb
        return tensor[bb]

    def __getitem__(self, index):
        raw = self._get_sample(index)

        if self.raw_transform is not None:
            raw = self.raw_transform(raw)

        if self.transform is not None:
            raw = self.transform(raw)
            if isinstance(raw, list):
                assert len(raw) == 1
                raw = raw[0]

            if self.trafo_halo is not None:
                raw = self.crop(raw)

        raw = ensure_tensor_with_channels(raw, ndim=self._ndim, dtype=self.dtype)
        if self.augmentations is not None:
            aug1, aug2 = self.augmentations
            raw1, raw2 = aug1(raw), aug2(raw)
            return raw1, raw2

        return raw

    # need to overwrite pickle to support h5py
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["raw"]
        return state

    def __setstate__(self, state):
        raw_path, raw_key = state["raw_path"], state["raw_key"]
        roi = state["roi"]
        try:
            raw = load_data(raw_path, raw_key)
            if roi is not None:
                raw = RoiWrapper(raw, (slice(None),) + roi) if state["_with_channels"] else RoiWrapper(raw, roi)
            state["raw"] = raw
        except Exception:
            msg = f"RawDataset could not be deserialized because of missing {raw_path}, {raw_key}.\n"
            msg += "The dataset is deserialized in order to allow loading trained models from a checkpoint.\n"
            msg += "But it cannot be used for further training and wil throw an error."
            warnings.warn(msg)
            state["raw"] = None

        self.__dict__.update(state)
