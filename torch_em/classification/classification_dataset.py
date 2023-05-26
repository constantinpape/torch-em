import numpy as np
import torch
from skimage.transform import resize


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, data, target, normalization, augmentation, image_shape):
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
