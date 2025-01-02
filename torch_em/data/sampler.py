import numpy as np
from typing import List, Optional, Callable, Union


class MinForegroundSampler:
    def __init__(self, min_fraction: float, background_id: int = 0, p_reject: float = 1.0):
        self.min_fraction = min_fraction
        self.background_id = background_id
        self.p_reject = p_reject

    def __call__(self, x, y=None):
        # we do this so it's also possible to use the MinForegroundSampler
        # for raw data, in order to filter out not imaged areas, for example in
        # large EM volumes.
        if y is None:
            y = x

        size = float(y.size)
        if isinstance(self.background_id, int):
            foreground_fraction = np.sum(y != self.background_id) / size
        else:
            foreground_fraction = np.sum(np.logical_not(np.isin(y, self.background_id))) / size

        if foreground_fraction > self.min_fraction:
            return True
        else:
            return np.random.rand() > self.p_reject


class MinSemanticLabelForegroundSampler:
    def __init__(
        self, semantic_ids: List[int], min_fraction: float, min_fraction_per_id: bool = False, p_reject: float = 1.0
    ):
        self.semantic_ids = semantic_ids
        self.min_fraction = min_fraction
        self.p_reject = p_reject
        self.min_fraction_per_id = min_fraction_per_id

    def __call__(self, x, y):
        size = float(y.size)

        if self.min_fraction_per_id:
            foreground_fraction = [np.sum(np.isin(y, idx)) / size for idx in self.semantic_ids]
        else:
            foreground_fraction = [np.sum(np.isin(y, self.semantic_ids))]

        if all(foreground_fraction) > self.min_fraction:
            return True
        else:
            return np.random.rand() > self.p_reject


class MinIntensitySampler:
    def __init__(self, min_intensity: int, function: Union[str, Callable] = "median", p_reject: float = 1.0):
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
    def __init__(self, min_num_instances: int = 2, p_reject: float = 1.0, min_size: Optional[int] = None):
        self.min_num_instances = min_num_instances
        self.p_reject = p_reject
        self.min_size = min_size

    def __call__(self, x, y):
        uniques, sizes = np.unique(y, return_counts=True)
        if self.min_size is not None:
            filter_ids = uniques[sizes >= self.min_size]
            uniques = filter_ids

        if len(uniques) >= self.min_num_instances:
            return True
        else:
            return np.random.rand() > self.p_reject


class MinTwoInstanceSampler:
    # for the case of min_num_instances=2 this is roughly 10x faster
    # than using MinInstanceSampler since np.unique is slow
    def __init__(self, p_reject: float = 1.0):
        self.p_reject = p_reject

    def __call__(self, x, y):
        sample_value = y.flat[0]
        if (y != sample_value).any():
            return True
        else:
            return np.random.rand() > self.p_reject


class MinNoToBackgroundBoundarySampler:
    # Sometimes it is necessary to ignore boundaries to the background
    # in RF training. Then, it can happen that even with 2 instances in the
    # image while sampling there will be no boundary in the image after the
    # label_transform and the RF only learns one class (Error further downstream).
    # Therefore, this sampler is needed. Unfortunatley, the NoToBackgroundBoundaryTransform
    # is then calculated multiple times.
    def __init__(self, trafo: Callable, min_fraction: float = 0.01, p_reject: float = 1.0):
        self.trafo = trafo
        self.bg_label = trafo.bg_label
        self.mask_label = trafo.mask_label
        self.min_fraction = min_fraction
        self.p_reject = p_reject

    def __call__(self, x, y):
        y_boundaries = self.trafo(y)
        y_boundaries[y_boundaries == self.mask_label] = self.bg_label
        size = float(y_boundaries.size)
        foreground_fraction = np.sum(y_boundaries != self.bg_label) / size
        if foreground_fraction > self.min_fraction:
            return True
        else:
            return np.random.rand() > self.p_reject
