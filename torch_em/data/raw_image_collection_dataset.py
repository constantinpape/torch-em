import os
import numpy as np
from typing import List, Union, Tuple, Optional, Any, Callable

import torch

from ..util import ensure_tensor_with_channels, load_image, supports_memmap


class RawImageCollectionDataset(torch.utils.data.Dataset):
    max_sampling_attempts = 500

    def _check_inputs(self, raw_images, full_check):
        if not full_check:
            return

        is_multichan = None
        for raw_im in raw_images:

            # we only check for compatible shapes if images support memmap, because
            # we don't want to load everything into ram
            if supports_memmap(raw_im):
                shape = load_image(raw_im).shape
                assert len(shape) in (2, 3)

                multichan = len(shape) == 3
                if is_multichan is None:
                    is_multichan = multichan
                else:
                    assert is_multichan == multichan

                # we assume axis last
                if is_multichan:
                    shape = shape[:-1]

    def __init__(
        self,
        raw_image_paths: Union[List[Any], str, os.PathLike],
        patch_shape: Tuple[int, ...],
        raw_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        dtype: torch.dtype = torch.float32,
        n_samples: Optional[int] = None,
        sampler: Optional[Callable] = None,
        augmentations: Optional[Callable] = None,
        full_check: bool = False,
    ):
        self._check_inputs(raw_image_paths, full_check)
        self.raw_images = raw_image_paths
        self._ndim = 2

        assert len(patch_shape) == self._ndim
        self.patch_shape = patch_shape

        self.raw_transform = raw_transform
        self.transform = transform
        self.dtype = dtype
        self.sampler = sampler

        if n_samples is None:
            self._len = len(self.raw_images)
            self.sample_random_index = False
        else:
            self._len = n_samples
            self.sample_random_index = True

        if augmentations is not None:
            assert len(augmentations) == 2
        self.augmentations = augmentations

    def __len__(self):
        return self._len

    @property
    def ndim(self):
        return self._ndim

    def _sample_bounding_box(self, shape):
        bb_start = [
            np.random.randint(0, sh - psh) if sh - psh > 0 else 0 for sh, psh in zip(shape, self.patch_shape)
        ]
        return tuple(slice(start, start + psh) for start, psh in zip(bb_start, self.patch_shape))

    def _ensure_patch_shape(self, raw, have_raw_channels, channel_first):
        shape = raw.shape
        if have_raw_channels and channel_first:
            shape = shape[1:]

        if any(sh < psh for sh, psh in zip(shape, self.patch_shape)):
            pw = [(0, max(0, psh - sh)) for sh, psh in zip(shape, self.patch_shape)]

            if have_raw_channels and channel_first:
                pw_raw = [(0, 0), *pw]
            elif have_raw_channels and not channel_first:
                pw_raw = [*pw, (0, 0)]
            else:
                pw_raw = pw

            raw = np.pad(raw, pw_raw)
        return raw

    def _get_sample(self, index):
        if self.sample_random_index:
            index = np.random.randint(0, len(self.raw_images))

        raw = load_image(self.raw_images[index])
        have_raw_channels = raw.ndim == 3

        # We determine if the image has channels as the first or last axis based on the array shape.
        # This will work only for images with less than 16 channels!
        # If the last axis has a length smaller than 16 we assume that it is the channel axis,
        # otherwise we assume it is a spatial axis and that the first axis is the channel axis.
        channel_first = None
        if have_raw_channels:
            channel_first = raw.shape[-1] > 16

        raw = self._ensure_patch_shape(raw, have_raw_channels, channel_first)

        shape = raw.shape
        # we assume images are loaded with channel last!
        if have_raw_channels:
            shape = shape[:-1]

        # sample random bounding box for this image
        bb = self._sample_bounding_box(shape)
        raw = np.array(raw[bb])

        if self.sampler is not None:
            sample_id = 0
            while not self.sampler(raw):
                bb = self._sample_bounding_box(shape)
                raw = np.array(raw[bb])
                sample_id += 1
                if sample_id > self.max_sampling_attempts:
                    raise RuntimeError(f"Could not sample a valid batch in {self.max_sampling_attempts} attempts")

        # to channel first
        if have_raw_channels:
            raw = raw.transpose((2, 0, 1))

        return raw

    def __getitem__(self, index):
        raw = self._get_sample(index)

        if self.raw_transform is not None:
            raw = self.raw_transform(raw)

        if self.transform is not None:
            raw = self.transform(raw)
            assert len(raw) == 1
            raw = raw[0]
            # if self.trafo_halo is not None:
            #     raw = self.crop(raw)

        raw = ensure_tensor_with_channels(raw, ndim=self._ndim, dtype=self.dtype)
        if self.augmentations is not None:
            aug1, aug2 = self.augmentations
            raw1, raw2 = aug1(raw), aug2(raw)
            return raw1, raw2

        return raw
