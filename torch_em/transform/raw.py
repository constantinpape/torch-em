import numpy as np
import torch
from torchvision import transforms


#
# normalization functions
#


def standardize(raw, mean=None, std=None, axis=None, eps=1e-7):
    raw = raw.astype('float32')

    mean = raw.mean(axis=axis, keepdims=True) if mean is None else mean
    raw -= mean

    std = raw.std(axis=axis, keepdims=True) if std is None else std
    raw /= (std + eps)

    return raw


def normalize(raw, minval=None, maxval=None, axis=None, eps=1e-7):
    raw = raw.astype('float32')

    minval = raw.min(axis=axis, keepdims=True) if minval is None else minval
    raw -= minval

    maxval = raw.max(axis=axis, keepdims=True) if maxval is None else maxval
    raw /= (maxval + eps)

    return raw


def normalize_percentile(raw, lower=1.0, upper=99.0, axis=None, eps=1e-7):
    v_lower = np.percentile(raw, lower, axis=axis, keepdims=True)
    v_upper = np.percentile(raw, upper, axis=axis, keepdims=True) - v_lower
    return normalize(raw, v_lower, v_upper, eps=eps)


def normalize_torch(tensor, minval=None, maxval=None, eps=1e-7):
    tensor = tensor.type(torch.float32)

    minval = tensor.min() if minval is None else minval
    tensor -= minval

    maxval = tensor.max() if maxval is None else maxval
    tensor /= (maxval + eps)

    return tensor


# TODO
#
# intensity augmentations / noise augmentations
#
# modified from https://github.com/kreshuklab/spoco/blob/main/spoco/transforms.py
class RandomContrast():
    """
    Adjust contrast by scaling image to `mean + alpha * (image - mean)`.
    """
    def __init__(self, alpha=(0.5, 1.5), mean=0.0, clip_kwargs={}): # {'a_min': 0, 'a_max': 1}):
        self.alpha = alpha
        self.mean = mean
        self.clip_kwargs = clip_kwargs

    def __call__(self, img):
        alpha = np.random.uniform(self.alpha[0], self.alpha[1])
        result = self.mean + alpha * (img - self.mean)
        if self.clip_kwargs:
            return np.clip(result, **self.clip_kwargs)
        return result


class AdditiveGaussianNoise():
    """
    Add random Gaussian noise to image.
    """
    def __init__(self, scale=(0.0, 1.0)):
        self.scale = scale

    def __call__(self, img):
        std = np.random.uniform(self.scale[0], self.scale[1])
        gaussian_noise = np.random.normal(0, std, size=img.shape)
        return img + gaussian_noise


class AdditivePoissonNoise():
    """
    Add random Poisson noise to image.
    """
    def __init__(self, lam=(0.0, 1.0)):
        self.lam = lam

    def __call__(self, img):
        lam = np.random.uniform(self.lam[0], self.lam[1])
        poisson_noise = np.random.poisson(lam, size=img.shape)
        return img + poisson_noise


def get_default_mean_teacher_augmentations(p=0.5):
    return transforms.Compose([
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=p),
        transforms.RandomApply([AdditiveGaussianNoise()], p=p),
        transforms.RandomApply([AdditivePoissonNoise()], p=p),
        normalize_torch,
        transforms.RandomApply([RandomContrast(clip_kwargs={'a_min': 0, 'a_max': 1})], p=p),
    ])


#
# default transformation:
# apply intensity augmentations and normalize
#

class RawTransform:
    def __init__(self, normalizer, augmentation1=None, augmentation2=None):
        self.normalizer = normalizer
        self.augmentation1 = augmentation1
        self.augmentation2 = augmentation2

    def __call__(self, raw):
        if self.augmentation1 is not None:
            raw = self.augmentation1(raw)
        raw = self.normalizer(raw)
        if self.augmentation2 is not None:
            raw = self.augmentation2(raw)
        return raw


def get_raw_transform(normalizer=standardize, augmentation1=None, augmentation2=None):
    return RawTransform(normalizer,
                        augmentation1=augmentation1,
                        augmentation2=augmentation2)

