from typing import Sequence, Tuple

import numpy as np
import torch

from numpy.typing import ArrayLike
from skimage.transform import resize


class ClassificationDataset(torch.utils.data.Dataset):
    """Dataset for classification training.

    Args:
        data: The input data for classification. Expects a sequence of array-like data.
            The data can be two or three dimensional.
        target: The target data for classification. Expects a sequence of the same length as `data`.
            Each value in the sequence must be a scalar.
        normalization: The normalization function.
        augmentation: The augmentation function.
        image_shape: The target shape of the data. If given, each sample will be resampled to this size.
    """
    def __init__(
        self,
        data: Sequence[ArrayLike],
        target: Sequence[ArrayLike],
        normalization: callable,
        augmentation: callable,
        image_shape: Tuple[int, ...],
    ):
        if len(data) != len(target):
            raise ValueError(f"Length of data and target don't agree: {len(data)} != {len(target)}")
        self.data = data
        self.target = target
        self.normalization = normalization
        self.augmentation = augmentation
        self.image_shape = image_shape

    def __len__(self):
        return len(self.data)

    def resize(self, x):
        """@private
        """
        out = [resize(channel, self.image_shape, preserve_range=True)[None] for channel in x]
        return np.concatenate(out, axis=0)

    def __getitem__(self, index):
        x, y = self.data[index], self.target[index]

        # apply normalization
        if self.normalization is not None:
            x = self.normalization(x)

        # resize to sample shape if it was given
        if self.image_shape is not None:
            x = self.resize(x)

        # apply augmentations (if any)
        if self.augmentation is not None:
            _shape = x.shape
            # adds unwanted batch axis
            x = self.augmentation(x)[0][0]
            assert x.shape == _shape

        return x, y
