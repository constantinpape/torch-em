"""Helper functionality for empirically determining the largest batch size or patch shape
that fits into GPU memory for a given network.

The maximum is found by running a forward pass (prediction) with dummy data and increasing
the respective parameter until the GPU runs out of memory, using an exponential bracketing
followed by a binary search to converge in a logarithmic number of steps.
"""

import gc
import warnings
from typing import Any, Callable, Optional, Tuple, Union

import torch


def _is_oom_error(exc: BaseException) -> bool:
    """Check whether an exception was raised because the GPU ran out of memory."""
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _resolve_device(model: torch.nn.Module, device: Optional[Union[torch.device, str]]) -> torch.device:
    """Resolve the device and ensure it is a CUDA device (OOM cannot be detected on CPU)."""
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "compute_max_batch_size and compute_max_patch_shape require a CUDA device, because running out "
            f"of GPU memory is used as the termination signal, but got a model / device of type '{device.type}'. "
            "Move the model to a GPU and / or pass device='cuda'."
        )
    return device


def _resolve_in_channels(model: torch.nn.Module, in_channels: Optional[int]) -> int:
    """Resolve the number of input channels, falling back to the model's `in_channels` attribute."""
    if in_channels is None:
        in_channels = getattr(model, "in_channels", None)
    if in_channels is None:
        raise ValueError(
            "Could not determine the number of input channels from the model. Please pass 'in_channels' explicitly."
        )
    return int(in_channels)


def _resolve_min_divisible(
    model: torch.nn.Module, ndim: int, min_divisible: Optional[Tuple[int, ...]]
) -> Tuple[int, ...]:
    """Resolve the per-axis divisibility constraint, defaulting to (2 ** depth,) * ndim for U-Nets."""
    if min_divisible is None:
        depth = getattr(model, "depth", None)
        factor = 2 ** depth if depth is not None else 1
        min_divisible = (factor,) * ndim
    if len(min_divisible) != ndim:
        raise ValueError(f"min_divisible {min_divisible} does not match the number of dimensions {ndim}.")
    return tuple(int(d) for d in min_divisible)


def _attempt_forward(
    model: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    in_channels: int,
    batch_size: int,
    patch_shape: Tuple[int, ...],
    prediction_function: Optional[Callable[[Any], Any]],
) -> bool:
    """Run a single forward pass with dummy data and report whether it fit into memory.

    Returns True if the forward pass succeeded, False if the GPU ran out of memory.
    Any other (non-OOM) exception is re-raised, since it indicates a genuine problem
    such as an invalid patch shape rather than a memory limit.
    """
    inp = out = None
    try:
        inp = torch.empty((batch_size, in_channels, *patch_shape), dtype=dtype, device=device).normal_()
        with torch.no_grad():
            out = model(inp) if prediction_function is None else prediction_function(model, inp)
        torch.cuda.synchronize(device)  # Surface asynchronous / lazy OOM errors here instead of later.
        return True
    except Exception as exc:
        if _is_oom_error(exc):
            return False
        raise
    finally:
        del inp, out
        gc.collect()
        torch.cuda.empty_cache()


def _search_max_int(fits: Callable[[int], bool], upper_bound: int) -> int:
    """Find the largest integer in [1, upper_bound] for which `fits` returns True.

    Assumes monotonicity: if a value fits, every smaller value fits as well. First an upper
    bracket is found by doubling, then the exact value is determined by binary search.
    The return value equals `upper_bound` exactly if the bound was reached without a failure.
    """
    if upper_bound < 1:
        raise ValueError(f"upper_bound must be a positive integer, got {upper_bound}.")
    if not fits(1):
        raise RuntimeError(
            "The model does not fit into memory even for the smallest configuration (batch size 1 / the smallest "
            "patch shape). Reduce the patch shape, use a smaller model or run on a device with more memory."
        )

    # Phase 1: exponential bracketing - double the candidate until it fails or exceeds the bound.
    last_ok = 1
    candidate = 2
    while candidate <= upper_bound and fits(candidate):
        last_ok = candidate
        candidate *= 2

    if candidate > upper_bound:
        # We never observed a failure within the bound. Check the bound itself.
        if last_ok == upper_bound or fits(upper_bound):
            return upper_bound
        first_fail = upper_bound
    else:
        # The loop terminated because `fits(candidate)` returned False.
        first_fail = candidate

    # Phase 2: binary search in the open interval (last_ok, first_fail).
    lo, hi = last_ok, first_fail
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid
    return lo


def compute_max_batch_size(
    model: torch.nn.Module,
    patch_shape: Tuple[int, ...],
    in_channels: Optional[int] = None,
    device: Optional[Union[torch.device, str]] = None,
    dtype: torch.dtype = torch.float32,
    safety_factor: float = 0.9,
    max_batch_size: int = 1024,
    prediction_function: Optional[Callable[[Any], Any]] = None,
) -> int:
    """Empirically determine the largest batch size that fits into GPU memory for a fixed patch shape.

    The batch size is increased (forward pass with dummy data, exponential bracketing followed by
    binary search) until the GPU runs out of memory. This requires a CUDA device, since running out
    of memory is used as the termination signal.

    Args:
        model: The model.
        patch_shape: The spatial shape of a single sample, without batch or channel axis,
            e.g. (512, 512) for 2D or (64, 128, 128) for 3D.
        in_channels: The number of input channels. By default this is derived from the model's
            'in_channels' attribute.
        device: The device of the model. If not given, will be derived from the model parameters.
            Must be a CUDA device.
        dtype: The data type of the dummy input data.
        safety_factor: Factor in (0, 1] applied to the empirically determined maximum, to stay clear
            of the out-of-memory boundary. The returned batch size is at least one.
        max_batch_size: The upper bound for the search. If the model does not run out of memory at this
            batch size, it is returned and a warning is issued (the true maximum may be larger).
        prediction_function: A wrapper function for prediction to enable custom prediction procedures.

    Returns:
        The maximum batch size.
    """
    device = _resolve_device(model, device)
    in_channels = _resolve_in_channels(model, in_channels)

    was_training = model.training
    model.eval()
    try:
        def fits(batch_size: int) -> bool:
            return _attempt_forward(
                model, device, dtype, in_channels, batch_size, patch_shape, prediction_function
            )

        max_fitting = _search_max_int(fits, max_batch_size)
    finally:
        model.train(was_training)

    if max_fitting >= max_batch_size:
        warnings.warn(
            f"The batch size search reached the upper bound 'max_batch_size'={max_batch_size} without running "
            "out of memory. The true maximum may be larger; increase 'max_batch_size' to search further."
        )
        return max_fitting

    return max(1, int(max_fitting * safety_factor))


def compute_max_patch_shape(
    model: torch.nn.Module,
    ndim: int,
    batch_size: int = 1,
    min_divisible: Optional[Tuple[int, ...]] = None,
    in_channels: Optional[int] = None,
    device: Optional[Union[torch.device, str]] = None,
    dtype: torch.dtype = torch.float32,
    safety_factor: float = 0.9,
    max_scale_factor: int = 128,
    prediction_function: Optional[Callable[[Any], Any]] = None,
) -> Tuple[int, ...]:
    """Empirically determine the largest patch shape that fits into GPU memory for a fixed batch size.

    The patch shape is grown isotropically as integer multiples of `min_divisible`, i.e. the candidate
    shapes are `(k * d0, k * d1, ...)` for increasing `k`, so that the network's divisibility constraints
    are always satisfied. The multiplier is increased (forward pass with dummy data, exponential bracketing
    followed by binary search) until the GPU runs out of memory. This requires a CUDA device, since running
    out of memory is used as the termination signal.

    Args:
        model: The model.
        ndim: The number of spatial dimensions, i.e. 2 for a 2D and 3 for a 3D model.
        batch_size: The (fixed) batch size to use for the search.
        min_divisible: The factors each spatial axis must be divisible by, which also define the smallest
            patch shape and the increment of the search. By default this is derived from the model's 'depth'
            attribute as (2 ** depth,) * ndim (the constraint for a U-Net), falling back to (1,) * ndim.
        in_channels: The number of input channels. By default this is derived from the model's
            'in_channels' attribute.
        device: The device of the model. If not given, will be derived from the model parameters.
            Must be a CUDA device.
        dtype: The data type of the dummy input data.
        safety_factor: Factor in (0, 1] applied to the empirically determined maximum multiplier, to stay
            clear of the out-of-memory boundary. The returned patch shape is at least 'min_divisible'.
        max_scale_factor: The upper bound for the multiplier search. If the model does not run out of memory
            at this multiplier, the corresponding patch shape is returned and a warning is issued.
        prediction_function: A wrapper function for prediction to enable custom prediction procedures.

    Returns:
        The maximum patch shape, as a tuple of length 'ndim'.
    """
    min_divisible = _resolve_min_divisible(model, ndim, min_divisible)
    device = _resolve_device(model, device)
    in_channels = _resolve_in_channels(model, in_channels)

    def scale(k: int) -> Tuple[int, ...]:
        return tuple(k * d for d in min_divisible)

    was_training = model.training
    model.eval()
    try:
        def fits(k: int) -> bool:
            return _attempt_forward(
                model, device, dtype, in_channels, batch_size, scale(k), prediction_function
            )

        max_fitting = _search_max_int(fits, max_scale_factor)
    finally:
        model.train(was_training)

    if max_fitting >= max_scale_factor:
        warnings.warn(
            f"The patch shape search reached the upper bound 'max_scale_factor'={max_scale_factor} without "
            "running out of memory. The true maximum may be larger; increase 'max_scale_factor' to search further."
        )
        return scale(max_fitting)

    return scale(max(1, int(max_fitting * safety_factor)))
