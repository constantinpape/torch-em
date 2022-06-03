import numpy as np


class MinForegroundSampler:
    def __init__(self, min_fraction, background_id=0, p_reject=1.0):
        self.min_fraction = min_fraction
        self.background_id = background_id
        self.p_reject = p_reject

    def __call__(self, x, y):
        size = float(y.size)
        foreground_fraction = np.sum(y != self.background_id) / size
        if foreground_fraction > self.min_fraction:
            return True
        else:
            return np.random.rand() > self.p_reject


class MinIntensitySampler:
    def __init__(self, min_intensity, function="median", p_reject=1.0):
        self.min_intensity = min_intensity
        self.function = getattr(np, function) if isinstance(function, str) else function
        assert callable(self.function)
        self.p_reject = p_reject

    def __call__(self, x, y=None):
        intensity = self.function(x)
        if intensity > self.min_intensity:
            return True
        else:
            return np.random.rand() > self.p_reject


class MinInstanceSampler:
    def __init__(self, min_num_instances=2, p_reject=1.0):
        self.min_num_instances = min_num_instances
        self.p_reject = p_reject

    def __call__(self, x, y):
        uniques = np.unique(y)
        if len(uniques) >= self.min_num_instances:
            return True
        else:
            return np.random.rand() > self.p_reject


class MinTwoInstanceSampler:
    # for the case of min_num_instances=2 this is roughly 10x faster
    # than using MinInstanceSampler since np.unique is slow
    def __init__(self, p_reject=1.0):
        self.p_reject = p_reject

    def __call__(self, x, y):
        sample_value = y.flat[0]
        if (y != sample_value).any():
            return True
        else:
            return np.random.rand() > self.p_reject
