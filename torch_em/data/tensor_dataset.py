from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch

from .image_collection_dataset import ImageCollectionDataset


class TensorDataset(ImageCollectionDataset):
    """A dataset for in-memory images and segmentation labels.

    The images and labels may be either numpy arrays or tensors.

    Args:
        images: The list of images.
        labels: The list of label images.
        label_transform: Transformation applied to the label data of a sample,
            before applying augmentations via `transform`.
        label_transform2: Transformation applied to the label data of a sample,
            after applying augmentations via `transform`.
        transform: Transformation applied to both the raw data and label data of a sample.
            This can be used to implement data augmentations.
        dtype: The return data type of the raw data.
        label_dtype: The return data type of the label data.
        n_samples: The length of this dataset. If None, the length will be set to `len(raw_image_paths)`.
        sampler: Sampler for rejecting samples according to a defined criterion.
            The sampler must be a callable that accepts the raw data and label data (as numpy arrays) as input.
        with_padding: Whether to pad samples to `patch_shape` if their shape is smaller.
        with_channels: Whether the raw data has channels.
    """
    def __init__(
        self,
        images: List[Union[np.ndarray, torch.tensor]],
        labels: List[Union[np.ndarray, torch.tensor]],
        patch_shape: Tuple[int, ...],
        raw_transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
        label_transform2: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        dtype: torch.dtype = torch.float32,
        label_dtype: torch.dtype = torch.float32,
        n_samples: Optional[int] = None,
        sampler: Optional[Callable] = None,
        with_padding: bool = True,
        with_channels: bool = False,
    ) -> None:
        self.raw_images = images
        self.label_images = labels
        self.patch_shape = patch_shape
        self.with_channels = with_channels
        self._check_inputs()
        self._ndim = len(self.patch_shape)

        self.with_label_channels = False
        self.have_tensor_data = True

        self.raw_transform = raw_transform
        self.label_transform = label_transform
        self.label_transform2 = label_transform2
        self.transform = transform
        self.sampler = sampler
        self.with_padding = with_padding

        self.dtype = dtype
        self.label_dtype = label_dtype

        if n_samples is None:
            self._len = len(self.raw_images)
            self.sample_random_index = False
        else:
            self._len = n_samples
            self.sample_random_index = True

    def _check_inputs(self):
        ndim = len(self.patch_shape)
        if len(self.raw_images) != len(self.label_images):
            raise ValueError(
                f"Number of images and labels does not match: {len(self.raw_images)}, {len(self.label_images)}"
            )
        for image, labels in zip(self.raw_images, self.label_images):
            im_shape = image.shape
            if self.with_channels and len(im_shape) != ndim + 1:
                raise ValueError("Image shape does not match the patch shape")
            elif not self.with_channels and len(im_shape) != ndim:
                raise ValueError("Image shape does not match the patch shape")

            if self.with_channels and im_shape[1:] != labels.shape:
                raise ValueError("Image and label shape does not match")
            elif not self.with_channels and im_shape != labels.shape:
                raise ValueError("Image and label shape does not match")
