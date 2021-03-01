

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
# default transformation:
# apply intensity augmentations and normalize
#

# TODO noise augmentations
class RawTransform:
    def __init__(self, normalizer):
        self.normalizer = normalizer

    def __call__(self, raw):
        return self.normalizer(raw)


def get_raw_transform(normalizer=standardize):
    return RawTransform(normalizer)
