import numpy as np
from typing import List, Optional, Callable, Union


class MinForegroundSampler:
    """A sampler to reject samples with a low fraction of foreground pixels in the labels.

    Args:
        min_fraction: The minimal fraction of foreground pixels for accepting a sample.
        background_id: The id of the background label.
        p_reject: The probability for rejecting a sample that does not meet the criterion.
    """
    def __init__(self, min_fraction: float, background_id: int = 0, p_reject: float = 1.0):
        self.min_fraction = min_fraction
        self.background_id = background_id
        self.p_reject = p_reject

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> bool:
        """Check the sample.

        Args:
            x: The raw data.
            y: The label data.

        Returns:
            Whether to accept this sample.
        """
        # We do this so that it's also possible to use the MinForegroundSampler for raw data,
        # in order to filter out areas that are not imaged, for example for large EM volumes.
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
    """A sampler to reject samples with a low fraction of foreground pixels in the semantic labels.

    Args:
        semantic_ids: The ids for semantic classes to take into account.
        min_fraction: The minimal fraction of foreground pixels for accepting a sample.
        min_fraction_per_id: Whether the minimal fraction is applied on a per label basis.
        p_reject: The probability for rejecting a sample that does not meet the criterion.
    """
    def __init__(
        self, semantic_ids: List[int], min_fraction: float, min_fraction_per_id: bool = False, p_reject: float = 1.0
    ):
        self.semantic_ids = semantic_ids
        self.min_fraction = min_fraction
        self.p_reject = p_reject
        self.min_fraction_per_id = min_fraction_per_id

    def __call__(self, x: np.ndarray, y: np.ndarray) -> bool:
        """Check the sample.

        Args:
            x: The raw data.
            y: The label data.

        Returns:
            Whether to accept this sample.
        """
        size = float(y.size)

        if self.min_fraction_per_id:
            foreground_fraction = [np.sum(np.isin(y, idx)) / size for idx in self.semantic_ids]
        else:
            foreground_fraction = [np.sum(np.isin(y, self.semantic_ids))]

        if all(fraction > self.min_fraction for fraction in foreground_fraction):
            return True
        else:
            return np.random.rand() > self.p_reject


class MinIntensitySampler:
    """A sampler to reject samples with low intensity in the raw data.

    Args:
        min_intensity: The minimal intensity for accepting a sample.
        function: The function for computing the intensity of the raw data.
            Can either be a function or a name of a valid numpy atttribute.
            In the latter case the corresponding numpy function is used.
        p_reject: The probability for rejecting a sample that does not meet the criterion.
    """
    def __init__(self, min_intensity: int, function: Union[str, Callable] = "median", p_reject: float = 1.0):
        self.min_intensity = min_intensity
        self.function = getattr(np, function) if isinstance(function, str) else function
        assert callable(self.function)
        self.p_reject = p_reject

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> bool:
        """Check the sample.

        Args:
            x: The raw data.
            y: The label data.

        Returns:
            Whether to accept this sample.
        """
        intensity = self.function(x)
        if intensity > self.min_intensity:
            return True
        else:
            return np.random.rand() > self.p_reject


class MinInstanceSampler:
    """A sampler to reject samples with too few instances in the label data.

    Args:
        min_num_instances: The minimum number of instances for accepting a sample.
        p_reject: The probability for rejecting a sample that does not meet the criterion.
        min_size: The minimal size for instances to be taken into account.
        reject_ids: The ids to reject (i.e. not consider) for sampling a valid input.
    """
    def __init__(
        self,
        min_num_instances: int = 2,
        p_reject: float = 1.0,
        min_size: Optional[int] = None,
        reject_ids: Optional[List[int]] = None,
    ):
        self.min_num_instances = min_num_instances
        self.p_reject = p_reject
        self.min_size = min_size
        self.reject_ids = reject_ids

        if self.reject_ids is not None:
            assert isinstance(self.reject_ids, list)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> bool:
        """Check the sample.

        Args:
            x: The raw data.
            y: The label data.

        Returns:
            Whether to accept this sample.
        """
        uniques, sizes = np.unique(y, return_counts=True)

        if self.min_size is not None:
            filter_ids = uniques[sizes >= self.min_size]
            uniques = filter_ids

        if self.reject_ids is not None:
            uniques = [idx for idx in uniques if idx not in self.reject_ids]

        if len(uniques) >= self.min_num_instances:
            return True
        else:
            return np.random.rand() > self.p_reject


class MinTwoInstanceSampler:
    """A sampler to reject samples with less than two instances in the label data.

    This is ca. 10x faster than `MinInstanceSampler(min_num_instances=2)` that which uses np.unique, which is slow.

    Args:
        p_reject: The probability for rejecting a sample that does not meet the criterion.
    """
    def __init__(self, p_reject: float = 1.0):
        self.p_reject = p_reject

    def __call__(self, x: np.ndarray, y: np.ndarray) -> bool:
        """Check the sample.

        Args:
            x: The raw data.
            y: The label data.

        Returns:
            Whether to accept this sample.
        """
        sample_value = y.flat[0]
        if (y != sample_value).any():
            return True
        else:
            return np.random.rand() > self.p_reject


# Sometimes it is necessary to ignore boundaries to the background
# in RF training. Then, it can happen that even with 2 instances in the
# image while sampling there will be no boundary in the image after the
# label_transform and the RF only learns one class (Error further downstream).
# Therefore, this sampler is needed. Unfortunatley, the NoToBackgroundBoundaryTransform
# is then calculated multiple times.
class MinNoToBackgroundBoundarySampler:
    """A sampler to reject samples for training with pseudo labels.

    Args:
        trafo: The transformation.
        min_fraction: The minimal fraction for accepting a sample.
        p_reject: The probability for rejecting a sample that does not meet the criterion.
    """
    def __init__(self, trafo, min_fraction: float = 0.01, p_reject: float = 1.0):
        self.trafo = trafo
        self.bg_label = trafo.bg_label
        self.mask_label = trafo.mask_label
        self.min_fraction = min_fraction
        self.p_reject = p_reject

    def __call__(self, x: np.ndarray, y: np.ndarray) -> bool:
        """Check the sample.

        Args:
            x: The raw data.
            y: The label data.

        Returns:
            Whether to accept this sample.
        """
        y_boundaries = self.trafo(y)
        y_boundaries[y_boundaries == self.mask_label] = self.bg_label
        size = float(y_boundaries.size)
        foreground_fraction = np.sum(y_boundaries != self.bg_label) / size
        if foreground_fraction > self.min_fraction:
            return True
        else:
            return np.random.rand() > self.p_reject
