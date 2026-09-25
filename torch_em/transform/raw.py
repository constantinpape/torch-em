from typing import Union, Optional, Tuple, Dict, Callable

import numpy as np

import torch
from torchvision import transforms


#
# normalization functions
#


TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}
"""@private
"""


def cast(inpt: Union[np.ndarray, torch.tensor], typestring: torch.dtype):
    """@private
    """
    if torch.is_tensor(inpt):
        assert typestring in TORCH_DTYPES, f"{typestring} not in TORCH_DTYPES"
        return inpt.to(TORCH_DTYPES[typestring])
    return inpt.astype(typestring)


def standardize(
    raw: np.ndarray,
    mean: Optional[float] = None,
    std: Optional[float] = None,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    eps: float = 1e-7,
) -> np.ndarray:
    """Standardize the input data by subtracting its mean and dividing by its standard deviation.

    Args:
        raw: The input data.
        mean: The mean value. If None, it will be computed from the data.
        std: The standard deviation. If None, it will be computed from the data.
        axis: The axis along which to compute the mean and standard deviation.
        eps: The epsilon value for numerical stability.

    Returns:
        The standardized input data.
    """
    raw = cast(raw, "float32")
    mean = raw.mean(axis=axis, keepdims=True) if mean is None else mean
    raw -= mean

    std = raw.std(axis=axis, keepdims=True) if std is None else std
    raw /= (std + eps)
    return raw


def _normalize_torch(tensor, minval, maxval, axis, eps):
    if axis:  # torch returns torch.return_types.min or torch.return_types.max
        minval = torch.amin(tensor, dim=axis, keepdim=True) if minval is None else minval
        tensor -= minval

        maxval = torch.amax(tensor, dim=axis, keepdim=True) if maxval is None else maxval
        tensor /= (maxval + eps)

        return tensor

    # keepdim can only be used in combination with dim
    minval = tensor.min() if minval is None else minval
    tensor -= minval

    maxval = tensor.max() if maxval is None else maxval
    tensor /= (maxval + eps)

    return tensor


def normalize(
    raw: Union[torch.tensor, np.ndarray],
    minval: Optional[float] = None,
    maxval: Optional[float] = None,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    eps: float = 1e-7,
) -> np.ndarray:
    """Normalize the input data so that it is in range [0, 1].

    Args:
        raw: The input data.
        minval: The minimum data value. If None, it will be computed from the data.
        maxval: The maximum data value. If None, it will be computed from the data.
        axis: The axis along which to compute the min and max value.
        eps: The epsilon value for numerical stability.

    Returns:
        The normalized input data.
    """
    raw = cast(raw, "float32")
    if torch.is_tensor(raw):
        return _normalize_torch(raw, minval=minval, maxval=maxval, axis=axis, eps=eps)

    minval = raw.min(axis=axis, keepdims=True) if minval is None else minval
    raw -= minval

    maxval = raw.max(axis=axis, keepdims=True) if maxval is None else maxval
    raw /= (maxval + eps)
    return raw


def normalize_percentile(
    raw: np.ndarray,
    lower: float = 1.0,
    upper: float = 99.0,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    eps: float = 1e-7,
) -> np.ndarray:
    """Normalize the input data based on percentile values.

    Args:
        raw: The input data.
        lower: The lower percentile.
        upper: The upper percentile.
        axis: The axis along which to compute the percentiles.
        eps: The epsilon value for numerical stability.

    Returns:
        The normalized input data.
    """
    v_lower = np.percentile(raw, lower, axis=axis, keepdims=True)
    v_upper = np.percentile(raw, upper, axis=axis, keepdims=True) - v_lower
    return normalize(raw, v_lower, v_upper, eps=eps)


class RandomPercentileNormalization:
    """Normalize inputs with randomly sampled percentile bounds.

    By default, the lower and upper percentiles are sampled uniformly from
    ``lower_percentile_bounds`` and ``upper_percentile_bounds``. If no upper bounds are given,
    they are inferred by mirroring the lower bounds around 50. Normal (Gaussian) sampling can be
    enabled explicitly with ``distribution="normal"`` and
    ``distribution_kwargs={"mean": ..., "std": ...}``. The sampled percentile intensities are
    mapped to 0 and 1, and values outside them are clipped, so the output is always in ``[0, 1]``.

    Examples:
        Uniform sampling with the default percentile bounds and reproducible random draws:

        ```python
        normalization = RandomPercentileNormalization(seed=42)
        ```

        Normal sampling with explicit distribution parameters:

        ```python
        normalization = RandomPercentileNormalization(
            distribution="normal",
            distribution_kwargs={"mean": 2.0, "std": 1.0},
            seed=42,
        )
        ```

    Args:
        lower_percentile_bounds: Inclusive clipping bounds for the lower percentile.
        upper_percentile_bounds: Inclusive clipping bounds for the upper percentile. If None, the
            bounds are inferred by mirroring ``lower_percentile_bounds`` around 50.
        distribution: Sampling distribution for the percentiles. Supported values are
            ``"uniform"`` and ``"normal"``.
        distribution_kwargs: Parameters for normal sampling, which must contain exactly
            ``{"mean": ..., "std": ...}``. The upper percentile uses the mirrored normal
            distribution. Uniform sampling does not take additional parameters.
        rounding_decimals: Number of decimal places used to round sampled percentiles. Set to None
            to disable rounding.
        axis: Axes over which to compute the intensity percentiles.
        seed: Optional non-negative integer seed for reproducible sampling. NumPy integer types
            are also supported. Each DataLoader worker derives a distinct stream from this seed.
            By default, the global NumPy random state is used, which respects DataLoader worker seeding.
        eps: Epsilon used for numerical stability during normalization.
    """

    def __init__(
        self,
        lower_percentile_bounds: Tuple[float, float] = (0.0, 5.0),
        upper_percentile_bounds: Optional[Tuple[float, float]] = None,
        distribution: str = "uniform",
        distribution_kwargs: Optional[Dict[str, float]] = None,
        rounding_decimals: Optional[int] = 1,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        seed: Optional[int] = None,
        eps: float = 1e-7,
    ):
        lower_percentile_bounds = self._validate_bounds(lower_percentile_bounds, upper=False)
        if upper_percentile_bounds is None:
            upper_percentile_bounds = tuple(100.0 - bound for bound in reversed(lower_percentile_bounds))
        upper_percentile_bounds = self._validate_bounds(upper_percentile_bounds, upper=True)
        if distribution not in ("uniform", "normal"):
            raise ValueError("distribution must be 'uniform' or 'normal'.")

        if distribution == "uniform":
            if distribution_kwargs is not None:
                raise ValueError("Uniform sampling does not accept distribution_kwargs.")
        else:
            if not isinstance(distribution_kwargs, dict) or set(distribution_kwargs) != {"mean", "std"}:
                raise ValueError("Normal sampling requires exactly the distribution_kwargs 'mean' and 'std'.")
            mean, std = float(distribution_kwargs["mean"]), float(distribution_kwargs["std"])
            if not np.isfinite(mean) or not lower_percentile_bounds[0] <= mean <= lower_percentile_bounds[1]:
                raise ValueError("The normal distribution mean must be finite and within lower_percentile_bounds.")
            if not np.isfinite(std) or std < 0.0:
                raise ValueError("The normal distribution std must be finite and non-negative.")
            distribution_kwargs = {"mean": mean, "std": std}

        if rounding_decimals is not None and (
            not isinstance(rounding_decimals, int) or isinstance(rounding_decimals, bool) or rounding_decimals < 0
        ):
            raise ValueError("rounding_decimals must be a non-negative integer or None.")
        if not np.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be finite and greater than zero.")
        if seed is not None:
            if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
                raise TypeError("seed must be an integer or None.")
            if seed < 0:
                raise ValueError("seed must be non-negative.")
            seed = int(seed)

        self.lower_percentile_bounds = lower_percentile_bounds
        self.upper_percentile_bounds = upper_percentile_bounds
        self.distribution = distribution
        self.distribution_kwargs = distribution_kwargs
        self.rounding_decimals = rounding_decimals
        self.axis = axis
        self.seed = seed
        self.eps = float(eps)
        self._random_generator = None
        self._random_generator_worker_id = None

    @staticmethod
    def _validate_bounds(values, upper):
        name = "upper_percentile_bounds" if upper else "lower_percentile_bounds"
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError(f"{name} must contain exactly two values.")
        lower_bound, upper_bound = (float(value) for value in values)
        finite = np.isfinite(lower_bound) and np.isfinite(upper_bound)
        if upper:
            valid = 50.0 < lower_bound <= upper_bound <= 100.0
            interval = "(50, 100]"
        else:
            valid = 0.0 <= lower_bound <= upper_bound < 50.0
            interval = "[0, 50)"
        if not finite or not valid:
            raise ValueError(f"{name} must be a finite interval within {interval}.")
        return lower_bound, upper_bound

    def _round(self, value):
        return float(value) if self.rounding_decimals is None else round(float(value), self.rounding_decimals)

    def _get_random_generator(self):
        if self.seed is None:
            return np.random

        worker_info = torch.utils.data.get_worker_info()
        worker_id = None if worker_info is None else worker_info.id
        if self._random_generator is None or self._random_generator_worker_id != worker_id:
            seed_sequence = np.random.SeedSequence([self.seed, 0 if worker_id is None else worker_id])
            self._random_generator = np.random.default_rng(seed_sequence)
            self._random_generator_worker_id = worker_id
        return self._random_generator

    def sample_percentiles(self) -> Tuple[float, float]:
        """Sample and return a valid ``(lower, upper)`` percentile pair."""
        random_generator = self._get_random_generator()
        if self.distribution == "uniform":
            lower = random_generator.uniform(*self.lower_percentile_bounds)
            upper = random_generator.uniform(*self.upper_percentile_bounds)
        else:
            mean = self.distribution_kwargs["mean"]
            std = self.distribution_kwargs["std"]
            lower = mean if std == 0.0 else random_generator.normal(mean, std)
            upper = 100.0 - (mean if std == 0.0 else random_generator.normal(mean, std))

        # Normal distribution tails may leave the configured percentile interval.
        lower = float(np.clip(self._round(lower), *self.lower_percentile_bounds))
        upper = float(np.clip(self._round(upper), *self.upper_percentile_bounds))
        return lower, upper

    def __call__(self, raw: Union[np.ndarray, torch.tensor]) -> Union[np.ndarray, torch.tensor]:
        lower, upper = self.sample_percentiles()
        normalized = normalize_percentile(raw, lower=lower, upper=upper, axis=self.axis, eps=self.eps)
        if torch.is_tensor(normalized):
            return torch.clamp(normalized, min=0.0, max=1.0)
        return np.clip(normalized, 0.0, 1.0)


#
# Intensity Augmentations / Noise Augmentations.
#

# modified from https://github.com/kreshuklab/spoco/blob/main/spoco/transforms.py
class RandomContrast:
    """Transformation to adjust contrast by scaling image to `mean + alpha * (image - mean)`.

    Args:
        alpha: Minimal and maximal alpha value for adjusting the contrast.
            The value for the transformation will be drawn uniformly from the corresponding interval.
        mean: Mean value for the image data.
        clip_kwargs: Keyword arguments for clipping the data after the contrast augmentation.
    """
    def __init__(
        self, alpha: Tuple[float, float] = (0.5, 2), mean: float = 0.5, clip_kwargs: Dict = {"a_min": 0, "a_max": 1}
    ):
        self.alpha = alpha
        self.mean = mean
        self.clip_kwargs = clip_kwargs

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Apply the augmentation to data.

        Args:
            img: The input image.

        Returns:
            The transformed image.
        """
        alpha = np.random.uniform(self.alpha[0], self.alpha[1])
        result = self.mean + alpha * (img - self.mean)
        if self.clip_kwargs:
            return np.clip(result, **self.clip_kwargs)
        return result


class AdditiveGaussianNoise:
    """Transformation to add random Gaussian noise to image.

    Args:
        scale: Scale for the noise.
        clip_kwargs: Keyword arguments for clipping the data after the transformation.
    """
    def __init__(self, scale: Tuple[float, float] = (0.0, 0.3), clip_kwargs: Dict = {"a_min": 0, "a_max": 1}):
        self.scale = scale
        self.clip_kwargs = clip_kwargs

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Apply the augmentation to data.

        Args:
            img: The input image.

        Returns:
            The transformed image.
        """
        std = np.random.uniform(self.scale[0], self.scale[1])
        gaussian_noise = np.random.normal(0, std, size=img.shape)

        if self.clip_kwargs:
            return np.clip(img + gaussian_noise, 0, 1)

        return img + gaussian_noise


class AdditivePoissonNoise:
    """Transformation to add random additive Poisson noise to image.

    Args:
        lam: Lambda value for the Poisson transformation.
        clip_kwargs: Keyword arguments for clipping the data after the transformation.
    """
    # Not sure if Poisson noise like this does make sense for data that is already normalized
    def __init__(self, lam: Tuple[float, float] = (0.0, 0.1), clip_kwargs: Dict = {"a_min": 0, "a_max": 1}):
        self.lam = lam
        self.clip_kwargs = clip_kwargs

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Apply the augmentation to data.

        Args:
            img: The input image.

        Returns:
            The transformed image.
        """
        lam = np.random.uniform(self.lam[0], self.lam[1])
        poisson_noise = np.random.poisson(lam, size=img.shape) / lam
        if self.clip_kwargs:
            return np.clip(img + poisson_noise, 0, 1)
        return img + poisson_noise


class PoissonNoise:
    """Transformation to add random data-dependant Poisson noise to image.

    Args:
        multiplier: Multiplicative factors for deriving the lambda factor from the data.
            The factor used for the transformation will be uniformly sampled form the range of this parameter.
        clip_kwargs: Keyword arguments for clipping the data after the transformation.
    """
    def __init__(self, multiplier: Tuple[float, float] = (5.0, 10.0), clip_kwargs: Dict = {"a_min": 0, "a_max": 1}):
        self.multiplier = multiplier
        self.clip_kwargs = clip_kwargs

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Apply the augmentation to data.

        Args:
            img: The input image.

        Returns:
            The transformed image.
        """
        multiplier = np.random.uniform(self.multiplier[0], self.multiplier[1])
        offset = img.min()
        poisson_noise = np.random.poisson((img - offset) * multiplier)

        if isinstance(img, torch.Tensor):
            poisson_noise = torch.Tensor(poisson_noise)
        poisson_noise = poisson_noise / multiplier + offset

        if self.clip_kwargs:
            return np.clip(poisson_noise, **self.clip_kwargs)
        return poisson_noise


class GaussianBlur:
    """Transformation to blur the image with a randomly drawn sigma value.

    Args:
        sigma: The sigma value for the transformation.
            The value used in the transformation will be uniformly drawn from the range specified here.
    """
    def __init__(self, sigma: Tuple[float, float] = (0.0, 3.0)):
        self.sigma = sigma

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Apply the augmentation to data.

        Args:
            img: The input image.

        Returns:
            The transformed image.
        """
        # Sample the sigma value. Note that we switch the bounds to ensure zero is excluded from sampling.
        sigma = np.random.uniform(self.sigma[1], self.sigma[0])
        # Determine the kernel size based on the sigma value.
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)

        return transforms.GaussianBlur(kernel_size, sigma=sigma)(img)


#
# Default Transformation: Apply intensity augmentations and normalize.
#

class RawTransform:
    """The transformation for raw data during training.

    Args:
        normalizer: The normalization function.
        augmentation1: Intensity augmentation applied before the normalization.
        augmentation2: Intensity augmentation applied after the normalization.
    """
    def __init__(
        self, normalizer: Callable, augmentation1: Optional[Callable] = None, augmentation2: Optional[Callable] = None
    ):
        self.normalizer = normalizer
        self.augmentation1 = augmentation1
        self.augmentation2 = augmentation2

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        """Apply the raw transformation.

        Args:
            raw: The raw data.

        Returns:
            The transformed raw data.
        """
        if self.augmentation1 is not None:
            raw = self.augmentation1(raw)

        raw = self.normalizer(raw)

        if self.augmentation2 is not None:
            raw = self.augmentation2(raw)
        return raw


def get_raw_transform(
    normalizer: Callable = standardize,
    augmentation1: Optional[Callable] = None,
    augmentation2: Optional[Callable] = None
) -> Callable:
    """Get the raw transformation.

    Args:
        normalizer: The normalization function.
        augmentation1: Intensity augmentation applied before the normalization.
        augmentation2: Intensity augmentation applied after the normalization.

    Returns:
        The raw transformation.
    """
    return RawTransform(normalizer, augmentation1=augmentation1, augmentation2=augmentation2)


def get_default_mean_teacher_augmentations(
    p: float = 0.3,
    norm: Optional[Callable] = None,
    blur_kwargs: Optional[Dict] = None,
    poisson_kwargs: Optional[Dict] = None,
    gaussian_kwargs: Optional[Dict] = None,
) -> Callable:
    """Get the default augmentations for mean teacher training.

    The default values for the augmentations are designed for an image with pixel values in range [0, 1].
    By default, a normalization transformation is applied for this reason.

    Args:
        p: The probability for applying the individual intensity transformations.
        norm: The noromaization function.
        blur_kwargs: The keyword arguments for `GaussianBlur`.
        poisson_kwargs: The keyword arguments for `PoissonNoise`.
        gaussian_kwargs: The keyword arguments for `AdditiveGaussianNoise`.

    Returns:
        The raw transformation with augmentations.
    """
    if norm is None:
        norm = normalize

    aug1 = transforms.Compose([
        norm,
        transforms.RandomApply([GaussianBlur(**({} if blur_kwargs is None else blur_kwargs))], p=p),
        transforms.RandomApply([PoissonNoise(**({} if poisson_kwargs is None else poisson_kwargs))], p=p/2),
        transforms.RandomApply([AdditiveGaussianNoise(**({} if gaussian_kwargs is None else gaussian_kwargs))], p=p/2),
    ])

    aug2 = transforms.RandomApply([RandomContrast(clip_kwargs={"a_min": 0, "a_max": 1})], p=p)
    return get_raw_transform(normalizer=norm, augmentation1=aug1, augmentation2=aug2)
