import numpy as np


#
# normalization functions
#


def standardize(raw, mean=None, std=None, eps=1e-7):
    raw = raw.astype('float32')
    mean = raw.mean() if mean is None else mean
    raw -= mean
    std = raw.std() if std is None else std
    raw /= (std + eps)
    return raw


def normalize(raw, minval=None, maxval=None, eps=1e-7):
    raw = raw.astype('float32')
    minval = raw.min() if minval is None else minval
    raw -= minval
    maxval = raw.max() if maxval is None else maxval
    raw /= (maxval + eps)
    return raw


# TODO
def normalize_percentile():
    pass


# TODO
#
# intensity augmentations / noise augmentations
#


#
# defect augmentations
#


# TODO more defect types
class EMDefectAugmentation:
    def __init__(self, p_drop_slice):
        self.p_drop_slice = p_drop_slice

    def __call__(self, raw):
        for z in range(raw.shape[0]):
            if np.random.rand() < self.p_drop_slice:
                raw[z] = 0
        return raw


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
