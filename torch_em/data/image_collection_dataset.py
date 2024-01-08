import numpy as np
import torch
from ..util import (ensure_spatial_array, ensure_tensor_with_channels,
                    load_image, supports_memmap)


class ImageCollectionDataset(torch.utils.data.Dataset):
    max_sampling_attempts = 500
    max_sampling_attempts_image = 50

    def _check_inputs(self, raw_images, label_images, full_check):
        if len(raw_images) != len(label_images):
            raise ValueError(f"Expect same number of  and label images, got {len(raw_images)} and {len(label_images)}")

        if not full_check:
            return

        is_multichan = None
        for raw_im, label_im in zip(raw_images, label_images):

            # we only check for compatible shapes if both images support memmap, because
            # we don't want to load everything into ram
            if supports_memmap(raw_im) and supports_memmap(label_im):
                shape = load_image(raw_im).shape
                assert len(shape) in (2, 3)

                multichan = len(shape) == 3
                if is_multichan is None:
                    is_multichan = multichan
                else:
                    assert is_multichan == multichan

                if is_multichan:
                    # use heuristic to decide whether the data is stored in channel last or channel first order:
                    # if the last axis has a length smaller than 16 we assume that it's the channel axis,
                    # otherwise we assume it's a spatial axis and that the first axis is the channel axis.
                    if shape[-1] < 16:
                        shape = shape[:-1]
                    else:
                        shape = shape[1:]

                label_shape = load_image(label_im).shape
                if shape != label_shape:
                    msg = f"Expect raw and labels of same shape, got {shape}, {label_shape} for {raw_im}, {label_im}"
                    raise ValueError(msg)

    def __init__(
        self,
        raw_image_paths,
        label_image_paths,
        patch_shape,
        raw_transform=None,
        label_transform=None,
        label_transform2=None,
        transform=None,
        dtype=torch.float32,
        label_dtype=torch.float32,
        n_samples=None,
        sampler=None,
        full_check=False,
    ):
        self._check_inputs(raw_image_paths, label_image_paths, full_check=full_check)
        self.raw_images = raw_image_paths
        self.label_images = label_image_paths
        self._ndim = 2

        assert len(patch_shape) == self._ndim
        self.patch_shape = patch_shape

        self.raw_transform = raw_transform
        self.label_transform = label_transform
        self.label_transform2 = label_transform2
        self.transform = transform
        self.sampler = sampler

        self.dtype = dtype
        self.label_dtype = label_dtype

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
        bb_start = [
            np.random.randint(0, sh - psh) if sh - psh > 0 else 0
            for sh, psh in zip(shape, self.patch_shape)
        ]
        return tuple(slice(start, start + psh) for start, psh in zip(bb_start, self.patch_shape))

    def _ensure_patch_shape(self, raw, labels, have_raw_channels, have_label_channels, channel_first):
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

            # TODO: ensure padding for labels with channels, when supported (see `_get_sample` below)

            raw, labels = np.pad(raw, pw_raw), np.pad(labels, pw)
        return raw, labels

    def _load_data(self, raw_path, label_path):
        raw = load_image(raw_path, memmap=False)
        label = load_image(label_path, memmap=False)

        have_raw_channels = raw.ndim == 3
        have_label_channels = label.ndim == 3
        if have_label_channels:
            raise NotImplementedError("Multi-channel labels are not supported.")

        # We determine if the image has channels as the first or last axis based on the array shape.
        # This will work only for images with less than 16 channels!
        # If the last axis has a length smaller than 16 we assume that it is the channel axis,
        # otherwise we assume it is a spatial axis and that the first axis is the channel axis.
        channel_first = None
        if have_raw_channels:
            channel_first = raw.shape[-1] > 16

        raw, label = self._ensure_patch_shape(raw, label, have_raw_channels, have_label_channels, channel_first)
        shape = raw.shape

        prefix_box = tuple()
        if have_raw_channels:
            if channel_first:
                shape = shape[1:]
                prefix_box = (slice(None), )
            else:
                shape = shape[:-1]

        return raw, label, shape, prefix_box, have_raw_channels

    def _get_sample(self, index):
        if self.sample_random_index:
            index = np.random.randint(0, len(self.raw_images))

        # The filepath corresponding to this image.
        raw_path, label_path = self.raw_images[index], self.label_images[index]

        # Load the corresponding data.
        raw, label, shape, prefix_box, have_raw_channels = self._load_data(raw_path, label_path)

        # Sample random bounding box for this image.
        bb = self._sample_bounding_box(shape)
        raw_patch = np.array(raw[prefix_box + bb])
        label_patch = np.array(label[bb])

        if self.sampler is not None:
            sample_id = 0
            while not self.sampler(raw_patch, label_patch):
                bb = self._sample_bounding_box(shape)
                raw_patch = np.array(raw[prefix_box + bb])
                label_patch = np.array(label[bb])
                sample_id += 1

                # We need to avoid sampling from the same image over and over agagin,
                # otherwise this will fail just because of one or a few empty images.
                # Hence we update the image from which we sample sometimes.
                if sample_id % self.max_sampling_attempts_image == 0:
                    index = np.random.randint(0, len(self.raw_images))
                    raw_path, label_path = self.raw_images[index], self.label_images[index]
                    raw, label, shape, prefix_box, have_raw_channels = self._load_data(raw_path, label_path)

                if sample_id > self.max_sampling_attempts:
                    raise RuntimeError(f"Could not sample a valid batch in {self.max_sampling_attempts} attempts")

        # to channel first
        if have_raw_channels and len(prefix_box) == 0:
            raw_patch = raw_patch.transpose((2, 0, 1))

        return raw_patch, label_patch

    def __getitem__(self, index):
        raw, labels = self._get_sample(index)
        initial_label_dtype = labels.dtype

        if self.raw_transform is not None:
            raw = self.raw_transform(raw)

        if self.label_transform is not None:
            labels = self.label_transform(labels)

        if self.transform is not None:
            raw, labels = self.transform(raw, labels)
            # if self.trafo_halo is not None:
            #     raw = self.crop(raw)
            #     labels = self.crop(labels)

        # support enlarging bounding box here as well (for affinity transform) ?
        if self.label_transform2 is not None:
            labels = ensure_spatial_array(labels, self.ndim, dtype=initial_label_dtype)
            labels = self.label_transform2(labels)

        raw = ensure_tensor_with_channels(raw, ndim=self._ndim, dtype=self.dtype)
        labels = ensure_tensor_with_channels(labels, ndim=self._ndim, dtype=self.label_dtype)
        return raw, labels
