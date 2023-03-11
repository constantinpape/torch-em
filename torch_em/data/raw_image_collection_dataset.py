import numpy as np
import torch
from ..util import ensure_tensor_with_channels, load_image, supports_memmap


# TODO pad images that are too small for the patch shape
class RawImageCollectionDataset(torch.utils.data.Dataset):
    max_sampling_attempts = 500

    def _check_inputs(self, raw_images):
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
        raw_image_paths,
        patch_shape,
        raw_transform=None,
        transform=None,
        dtype=torch.float32,
        n_samples=None,
        sampler=None,
    ):
        self._check_inputs(raw_image_paths)
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

    def __len__(self):
        return self._len

    @property
    def ndim(self):
        return self._ndim

    def _sample_bounding_box(self, shape):
        if any(sh < psh for sh, psh in zip(shape, self.patch_shape)):
            raise NotImplementedError("Image padding is not supported yet.")
        bb_start = [
            np.random.randint(0, sh - psh) if sh - psh > 0 else 0
            for sh, psh in zip(shape, self.patch_shape)
        ]
        return tuple(slice(start, start + psh) for start, psh in zip(bb_start, self.patch_shape))

    def _get_sample(self, index):
        if self.sample_random_index:
            index = np.random.randint(0, len(self.raw_images))
        raw = load_image(self.raw_images[index])
        have_raw_channels = raw.ndim == 3

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
        return raw
