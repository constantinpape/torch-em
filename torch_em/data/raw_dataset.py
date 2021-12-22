import torch
import numpy as np
from elf.io import open_file
from elf.wrapper import RoiWrapper

from ..util import ensure_tensor_with_channels


class RawDataset(torch.utils.data.Dataset):
    """
    """
    max_sampling_attempts = 500

    @staticmethod
    def compute_len(path, key, patch_shape, with_channels):
        with open_file(path, mode="r") as f:
            shape = f[key].shape[1:] if with_channels else f[key].shape
        n_samples = int(np.prod(
            [float(sh / csh) for sh, csh in zip(shape, patch_shape)]
        ))
        return n_samples

    def __init__(
        self,
        raw_path,
        raw_key,
        patch_shape,
        raw_transform=None,
        transform=None,
        roi=None,
        dtype=torch.float32,
        n_samples=None,
        sampler=None,
        ndim=None,
        with_channels=False,
    ):
        self.raw_path = raw_path
        self.raw_key = raw_key
        self.raw = open_file(raw_path, mode="r")[raw_key]

        self._with_channels = with_channels
        if ndim is None:
            self._ndim = self.raw.ndim - 1 if with_channels else self.raw.ndim
        else:
            self._ndim = ndim
        assert self._ndim in (2, 3), "Invalid data dimensionality: {self._ndim}. Only 2d or 3d data is supported"

        if self._with_channels:
            assert self.raw.ndim == self._ndim + 1, f"{self.raw.ndim}, {self._ndim}"

        raw_ndim = self.raw.ndim - 1 if self._with_channels else self.raw.ndim
        if roi is not None:
            assert len(roi) == raw_ndim, f"{roi}, {raw_ndim}"
            self.raw = RoiWrapper(self.raw, (slice(None),) + roi) if self._with_channels else RoiWrapper(self.raw, roi)
        self.roi = roi
        self.shape = self.raw.shape[1:] if self._with_channels else self.raw.shape

        assert len(patch_shape) == raw_ndim, f"{patch_shape}, {raw_ndim}"
        self.patch_shape = patch_shape

        self.raw_transform = raw_transform
        self.transform = transform
        self.sampler = sampler
        self.dtype = dtype

        if n_samples is None:
            self._len = n_samples
        else:
            self._len = self.compute_len(raw_path, raw_key, self.patch_shape, with_channels)

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
        bb = self._sample_bounding_box()
        if self._with_channels:
            raw = self.raw[(slice(None),) + bb]
        else:
            raw = self.raw[bb]

        if self.sampler is not None:
            sample_id = 0
            while not self.sampler(raw):
                bb = self._sample_bounding_box()
                raw = self.raw[(slice(None),) + bb] if self._with_channels else self.raw[bb]
                sample_id += 1
                if sample_id > self.max_sampling_attempts:
                    raise RuntimeError(f"Could not sample a valid batch in {self.max_sampling_attempts} attempts")

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
            if self.trafo_halo is not None:
                raw = self.crop(raw)

        raw = ensure_tensor_with_channels(raw, ndim=self._ndim, dtype=self.dtype)
        return raw

    # need to overwrite pickle to support h5py
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["raw"]
        return state

    def __setstate__(self, state):
        state["raw"] = open_file(state["raw_path"], mode="r")[state["raw_key"]]
        self.__dict__.update(state)
