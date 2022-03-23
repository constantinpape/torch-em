import warnings

import torch
import numpy as np
from elf.io import open_file
from elf.wrapper import RoiWrapper

from ..util import ensure_spatial_array, ensure_tensor_with_channels


class SegmentationDataset(torch.utils.data.Dataset):
    """
    """
    max_sampling_attempts = 500

    @staticmethod
    def compute_len(path, key, patch_shape):
        with open_file(path, mode="r") as f:
            shape = f[key].shape
        n_samples = int(np.prod(
            [float(sh / csh) for sh, csh in zip(shape, patch_shape)]
        ))
        return n_samples

    def __init__(
        self,
        raw_path,
        raw_key,
        label_path,
        label_key,
        patch_shape,
        raw_transform=None,
        label_transform=None,
        label_transform2=None,
        transform=None,
        roi=None,
        dtype=torch.float32,
        label_dtype=torch.float32,
        n_samples=None,
        sampler=None,
        ndim=None,
        with_channels=False,
    ):
        self.raw_path = raw_path
        self.raw_key = raw_key
        self.raw = open_file(raw_path, mode="r")[raw_key]

        self.label_path = label_path
        self.label_key = label_key
        self.labels = open_file(label_path, mode="r")[label_key]

        self._with_channels = with_channels
        if ndim is None:
            self._ndim = self.labels.ndim
        else:
            self._ndim = ndim
        assert self._ndim in (2, 3), "Invalid data dimensionality: {self._ndim}. Only 2d or 3d data is supported"

        if self._with_channels:
            assert self.raw.shape[1:] == self.labels.shape, f"{self.raw.shape[1:]}, {self.labels.shape}"
            if self._ndim < self.labels.ndim:
                raise NotImplementedError("Using channels and smapling 2d slcies from 3d data is not supported yet")
        else:
            assert self.raw.shape == self.labels.shape, f"{self.raw.shape}, {self.labels.shape}"

        if roi is not None:
            assert len(roi) == self.labels.ndim, f"{roi}, {self.labels.ndim}"
            self.raw = RoiWrapper(self.raw, (slice(None),) + roi) if self._with_channels else RoiWrapper(self.raw, roi)
            self.labels = RoiWrapper(self.labels, roi)

        self.shape = self.labels.shape
        self.roi = roi

        assert len(patch_shape) == self.labels.ndim, f"{patch_shape}, {self.labels.ndim}"
        self.patch_shape = patch_shape

        self.raw_transform = raw_transform
        self.label_transform = label_transform
        self.label_transform2 = label_transform2
        self.transform = transform
        self.sampler = sampler

        self.dtype = dtype
        self.label_dtype = label_dtype

        self._len = self.compute_len(label_path, label_key, self.patch_shape) if n_samples is None else n_samples

        # TODO
        self.trafo_halo = None
        # self.trafo_halo = None if self.transform is None\
        #     else self.transform.halo(self.patch_shape)
        if self.trafo_halo is None:
            self.sample_shape = self.patch_shape
        else:
            if len(self.trafo_halo) == 2 and self._ndim == 3:
                self.trafo_halo = (0,) + self.trafo_halo
            assert len(self.trafo_halo) == self._ndim
            self.sample_shape = tuple(sh + ha for sh, ha in zip(self.patch_shape, self.trafo_halo))
            self.inner_bb = tuple(slice(ha, sh - ha) for sh, ha in zip(self.patch_shape, self.trafo_halo))

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
        if self.raw is None or self.labels is None:
            raise RuntimeError("SegmentationDataset has not been properly deserialized.")
        bb = self._sample_bounding_box()
        if self._with_channels:
            raw, labels = self.raw[(slice(None),) + bb], self.labels[bb]
        else:
            raw, labels = self.raw[bb], self.labels[bb]

        if self.sampler is not None:
            sample_id = 0
            while not self.sampler(raw, labels):
                bb = self._sample_bounding_box()
                if self._with_channels:
                    raw, labels = self.raw[(slice(None),) + bb], self.labels[bb]
                else:
                    raw, labels = self.raw[bb], self.labels[bb]
                sample_id += 1
                if sample_id > self.max_sampling_attempts:
                    raise RuntimeError(f"Could not sample a valid batch in {self.max_sampling_attempts} attempts")

        return raw, labels

    def crop(self, tensor):
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
        try:
            state["raw"] = open_file(raw_path, mode="r")[raw_key]
        except Exception:
            msg = f"SegmentationDataset could not be deserialized because of missing {raw_path}, {raw_key}.\n"
            msg += "The dataset is deserialized in order to allow loading trained models from a checkpoint.\n"
            msg += "But it cannot be used for further training and wil throw an error."
            warnings.warn(msg)
            state["raw"] = None
        try:
            state["labels"] = open_file(label_path, mode="r")[label_key]
        except Exception:
            msg = f"SegmentationDataset could not be deserialized because of missing {label_path}, {label_key}.\n"
            msg += "The dataset is deserialized in order to allow loading trained models from a checkpoint.\n"
            msg += "But it cannot be used for further training and wil throw an error."
            warnings.warn(msg)
            state["labels"] = None
        self.__dict__.update(state)
