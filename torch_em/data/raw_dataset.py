import os
import warnings
import numpy as np
from typing import List, Union, Tuple, Optional, Any

import torch

from elf.wrapper import RoiWrapper

from ..util import ensure_tensor_with_channels, ensure_patch_shape, load_data


class RawDataset(torch.utils.data.Dataset):
    """
    """
    max_sampling_attempts = 500

    @staticmethod
    def compute_len(shape, patch_shape):
        n_samples = int(np.prod([float(sh / csh) for sh, csh in zip(shape, patch_shape)]))
        return n_samples

    def __init__(
        self,
        raw_path: Union[List[Any], str, os.PathLike],
        raw_key: str,
        patch_shape: Tuple[int, ...],
        raw_transform=None,
        transform=None,
        roi: Optional[dict] = None,
        dtype: torch.dtype = torch.float32,
        n_samples: Optional[int] = None,
        sampler=None,
        ndim: Optional[int] = None,
        with_channels: bool = False,
        augmentations=None,
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
                raw=raw,
                labels=None,
                patch_shape=self.patch_shape,
                have_raw_channels=self._with_channels
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
