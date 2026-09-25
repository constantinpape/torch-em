import queue
import threading
import warnings
from copy import deepcopy
from concurrent import futures
from typing import Tuple, Union, Callable, Any, List, Optional

import numpy as np
import bioimage_cpp as bic
import torch
from numpy.typing import ArrayLike

try:
    from napari.utils import progress as tqdm
except ImportError:
    from tqdm import tqdm

from ..transform.raw import standardize


def predict_with_padding(
    model: torch.nn.Module,
    input_: np.ndarray,
    min_divisible: Tuple[int, ...],
    device: Optional[Union[torch.device, str]] = None,
    with_channels: bool = False,
    prediction_function: Callable[[Any], Any] = None
) -> np.ndarray:
    """Run prediction with padding for a model that can only deal with inputs divisible by specific factors.

    Args:
        model: The model.
        input_: The input for prediction.
        min_divisible: The minimal factors the input shape must be divisible by.
            For example, (16, 16) for a model that needs 2D inputs divisible by at least 16 pixels.
        device: The device of the model. If not given, will be derived from the model parameters.
        with_channels: Whether the input data contains channels.
        prediction_function: A wrapper function for prediction to enable custom prediction procedures.

    Returns:
        np.ndarray: The ouptut of the model.
    """
    if with_channels:
        assert len(min_divisible) + 1 == input_.ndim, f"{min_divisible}, {input_.ndim}"
        min_divisible_ = (1,) + min_divisible
    else:
        assert len(min_divisible) == input_.ndim
        min_divisible_ = min_divisible

    if any(sh % md != 0 for sh, md in zip(input_.shape, min_divisible_)):
        pad_width = tuple(
            (0, 0 if sh % md == 0 else md - sh % md)
            for sh, md in zip(input_.shape, min_divisible_)
        )
        crop_padding = tuple(slice(0, sh) for sh in input_.shape)
        input_ = np.pad(input_, pad_width, mode="reflect")
    else:
        crop_padding = None

    ndim = input_.ndim
    ndim_model = 1 + ndim if with_channels else 2 + ndim

    if device is None:
        device = next(model.parameters()).device

    expand_dim = (None,) * (ndim_model - ndim)
    with torch.no_grad():
        model_input = torch.from_numpy(input_[expand_dim]).to(device)
        output = model(model_input) if prediction_function is None else prediction_function(model, model_input)
        output = output.cpu().numpy()

    if crop_padding is not None:
        crop_padding = (slice(None),) * (output.ndim - len(crop_padding)) + crop_padding
        output = output[crop_padding]

    return output


def _pad_for_shift_left(arr, pad_vox, with_channels, mode="constant", constant_values=0.0):
    pad_left = tuple(pad_vox)
    pad_right = tuple(0 for _ in pad_vox)

    pad_width = tuple((pl, pr) for pl, pr in zip(pad_left, pad_right))
    if with_channels:
        pad_width = ((0, 0),) + pad_width

    arr_pad = np.pad(arr, pad_width, mode=mode, constant_values=constant_values)
    return arr_pad, pad_left


def _crop_after_shift_left(arr, pad_left, with_channels, original_shape_spatial):
    starts = pad_left
    stops = tuple(st + sh for st, sh in zip(starts, original_shape_spatial))
    spatial_slices = tuple(slice(st, sp) for st, sp in zip(starts, stops))
    return arr[(slice(None),) + spatial_slices] if with_channels else arr[spatial_slices]


def _load_block(input_, offset, block_shape, halo, padding_mode="reflect", with_channels=False):
    shape = input_.shape
    if with_channels:
        shape = shape[1:]

    starts = [off - ha for off, ha in zip(offset, halo)]
    stops = [off + bs + ha for off, bs, ha in zip(offset, block_shape, halo)]

    # we pad the input volume if necessary
    pad_left = None
    pad_right = None

    # check for padding to the left
    if any(start < 0 for start in starts):
        pad_left = tuple(abs(start) if start < 0 else 0 for start in starts)
        starts = [max(0, start) for start in starts]

    # check for padding to the right
    if any(stop > shape[i] for i, stop in enumerate(stops)):
        pad_right = tuple(stop - shape[i] if stop > shape[i] else 0 for i, stop in enumerate(stops))
        stops = [min(shape[i], stop) for i, stop in enumerate(stops)]

    bb = tuple(slice(start, stop) for start, stop in zip(starts, stops))
    if with_channels:
        data = input_[(slice(None),) + bb]
    else:
        data = input_[bb]

    ndim = len(shape)
    # pad if necessary
    if pad_left is not None or pad_right is not None:
        pad_left = (0,) * ndim if pad_left is None else pad_left
        pad_right = (0,) * ndim if pad_right is None else pad_right
        pad_width = tuple((pl, pr) for pl, pr in zip(pad_left, pad_right))
        if with_channels:
            pad_width = ((0, 0),) + pad_width
        data = np.pad(data, pad_width, mode=padding_mode)

        # extend the bounding box for downstream
        bb = tuple(
            slice(b.start - pl, b.stop + pr)
            for b, pl, pr in zip(bb, pad_left, pad_right)
        )

    return data, bb


def predict_with_halo(
    input_: ArrayLike,
    model: torch.nn.Module,
    gpu_ids: List[Union[str, int]],
    block_shape: Tuple[int, ...],
    halo: Tuple[int, ...],
    output: Optional[Union[ArrayLike, List[Tuple[ArrayLike, slice]]]] = None,
    preprocess: Callable[[Union[torch.Tensor, np.ndarray]], Union[torch.Tensor, np.ndarray]] = standardize,
    postprocess: Callable[[np.ndarray], np.ndarray] = None,
    with_channels: bool = False,
    skip_block: Callable[[Any], bool] = None,
    mask: Optional[ArrayLike] = None,
    disable_tqdm: bool = False,
    tqdm_desc: str = "predict with halo",
    prediction_function: Optional[Callable] = None,
    roi: Optional[Tuple[slice]] = None,
    iter_list: Optional[List[int]] = None,
    grid_shift: Optional[Tuple[float, ...]] = None,
) -> ArrayLike:
    """Run block-wise network prediction with a halo.

    Args:
        input_: The input data, can be a numpy array, a hdf5/zarr/z5py dataset or similar
        model: The network.
        gpu_ids: List of device ids to use for prediction. To run prediction on the CPU, pass `["cpu"]`.
        block_shape: The shape of the inner block to use for prediction.
        halo: The shape of the halo to use for prediction
        output: The output data, will be allocated if None is passed.
            Instead of a single output, this can also be a list of outputs and a slice for the corresponding channel.
        preprocess: Function to preprocess input data before passing it to the network.
        postprocess: Function to postprocess the network predictions.
        with_channels: Whether the input has a channel axis.
        skip_block: Function to evaluate whether a given input block will be skipped.
        mask: Elements outside the mask will be ignored in the prediction.
        disable_tqdm: Flag that allows to disable tqdm output (e.g. if function is called multiple times).
        tqdm_desc: Fescription shown by the tqdm output.
        prediction_function: A wrapper function for prediction to enable custom prediction procedures.
        roi: A region of interest of the input for which to run prediction.
        iter_list: Optional list of block ids to iterate over.
        grid_shift: Per-axis fractional shift of the grid in units of the block size. E.g. (0, 0.25, 0).
    Returns:
        The model output.
    """
    devices = [torch.device(gpu) for gpu in gpu_ids]
    models = [
        (model if next(model.parameters()).device == device else deepcopy(model).to(device), device)
        for device in devices
    ]
    n_workers = len(gpu_ids)

    # original shape (spatial only)
    shape0 = input_.shape
    shape_spatial0 = shape0[1:] if with_channels else shape0
    ndim = len(shape_spatial0)
    assert len(block_shape) == len(halo) == ndim

    # apply grid_shift via padding+cropping (zero padding)
    input_eff = input_
    mask_eff = mask

    if grid_shift is not None:
        assert len(grid_shift) == ndim, "grid_shift must match number of spatial dims"
        pad_vox = tuple(int(np.rint(abs(gs) * bs)) for gs, bs in zip(grid_shift, block_shape))

        if not isinstance(input_eff, np.ndarray):
            raise TypeError("grid_shift padding currently requires input_ to be a numpy array")

        input_eff, pad_left = _pad_for_shift_left(input_eff, pad_vox, with_channels=with_channels, mode="constant",
                                                  constant_values=0)

        if mask_eff is not None:
            if not isinstance(mask_eff, np.ndarray):
                raise TypeError("grid_shift padding currently requires mask to be a numpy array")
            mask_eff, _ = _pad_for_shift_left(mask_eff, pad_vox, with_channels=False, mode="constant",
                                              constant_values=0)
    else:
        pad_left = (0,) * ndim

    # shapes after shift-padding
    shape_eff = input_eff.shape
    shape_spatial_eff = shape_eff[1:] if with_channels else shape_eff

    # blocking (on the padded input)
    if roi is None:
        blocking = bic.utils.Blocking([0] * ndim, list(shape_spatial_eff), block_shape)
    else:
        assert len(roi) == ndim
        blocking_start = [0 if ro.start is None else ro.start for ro in roi]
        blocking_stop = [sh if ro.stop is None else ro.stop for ro, sh in zip(roi, shape_spatial_eff)]
        blocking = bic.utils.Blocking(blocking_start, blocking_stop, block_shape)

    # output allocation (for padded shape)
    if output is None:
        n_out = models[0][0].out_channels
        output = np.zeros((n_out,) + tuple(shape_spatial_eff), dtype="float32")
    elif grid_shift:
        raise ValueError(
            "grid_shift is not supported together with a user-provided `output`, because "
            "grid_shift requires internal zero-padding and a final cropping step. "
            "Pass `output=None` (let this function allocate the output) or disable `grid_shift`. "
            "Or pad the input manually beforehand."
        )

    def predict_block(block_id):
        worker_id = block_id % n_workers
        net, device = models[worker_id]

        with torch.no_grad():
            block = blocking.get_block(block_id)
            offset = [beg for beg in block.begin]
            inner_bb = tuple(slice(ha, ha + bs) for ha, bs in zip(halo, block.shape))

            if mask_eff is not None:
                mask_block, _ = _load_block(mask_eff, offset, block_shape, halo, with_channels=False)
                mask_block = mask_block[inner_bb].astype("bool")
                if mask_block.sum() == 0:
                    return

            inp, _ = _load_block(input_eff, offset, block_shape, halo, with_channels=with_channels)

            if skip_block is not None and skip_block(inp):
                return

            if preprocess is not None:
                inp = preprocess(inp)

            # add (channel) and batch axis
            expand_dims = np.s_[None] if with_channels else np.s_[None, None]
            inp = torch.from_numpy(inp[expand_dims]).to(device)

            prediction = net(inp) if prediction_function is None else prediction_function(net, inp)

            # allow for list of tensors
            try:
                prediction = prediction.cpu().numpy().squeeze(0)
            except AttributeError:
                prediction = prediction[0]
                prediction = prediction.cpu().numpy().squeeze(0)

            if postprocess is not None:
                prediction = postprocess(prediction)

            if prediction.ndim == ndim + 1:
                inner_bb_pred = (slice(None),) + inner_bb
            else:
                inner_bb_pred = inner_bb
            prediction = prediction[inner_bb_pred]

            if mask_eff is not None:
                if prediction.ndim == ndim + 1:
                    mb = np.broadcast_to(mask_block[None], prediction.shape)
                else:
                    mb = mask_block
                prediction[~mb] = 0

            bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
            if isinstance(output, list):  # we have multiple outputs and split the prediction channels
                for out, channel_slice in output:
                    this_bb = bb if out.ndim == ndim else (slice(None),) + bb
                    out[this_bb] = prediction[channel_slice]
            else:  # we only have a single output array
                if output.ndim == ndim + 1:
                    bb = (slice(None),) + bb
                output[bb] = prediction

    n_blocks = blocking.number_of_blocks
    iteration_ids = range(n_blocks) if iter_list is None else np.array(iter_list)

    with futures.ThreadPoolExecutor(n_workers) as tp:
        list(tqdm(tp.map(predict_block, iteration_ids),
                  total=len(iteration_ids),
                  disable=disable_tqdm,
                  desc=tqdm_desc))

    # crop away the shift padding so the returned output matches original shape
    if grid_shift is not None:
        output = _crop_after_shift_left(output, pad_left, with_channels=(output.ndim == ndim+1),
                                        original_shape_spatial=tuple(shape_spatial0))

    return output


# Sentinel returned by _prepare_block_input when a block should be skipped.
_SKIP = object()
# Sentinel pushed onto the pipeline queues to signal end-of-stream.
_STOP = object()


class _Aborted(Exception):
    """@private
    Raised inside worker threads to unwind cleanly once `stop_event` is set."""


class _AtomicCounter:
    """@private
    Lock-guarded integer counter used for sentinel reference counting."""

    def __init__(self, value: int):
        self._value = value
        self._lock = threading.Lock()

    def decrement(self) -> int:
        with self._lock:
            self._value -= 1
            return self._value


class _BlockJob:
    """@private
    A unit of work travelling through the prediction pipeline."""

    __slots__ = ("block", "inner_bb", "mask_block", "tensor", "prediction")

    def __init__(self, block, inner_bb, mask_block, tensor):
        self.block = block
        self.inner_bb = inner_bb
        self.mask_block = mask_block
        self.tensor = tensor  # CPU tensor [1, (C,) *spatial]; cleared after prediction
        self.prediction = None  # filled by the consumer


def _safe_get(q, stop_event, timeout=0.2):
    """@private
    Queue.get that aborts (raises _Aborted) once stop_event is set, to avoid deadlocks."""
    while not stop_event.is_set():
        try:
            return q.get(timeout=timeout)
        except queue.Empty:
            continue
    raise _Aborted()


def _safe_put(q, item, stop_event, timeout=0.2):
    """@private
    Queue.put that aborts (raises _Aborted) once stop_event is set, to avoid deadlocks."""
    while not stop_event.is_set():
        try:
            q.put(item, timeout=timeout)
            return
        except queue.Full:
            continue
    raise _Aborted()


def _prepare_block_input(input_, mask, block, block_shape, halo, with_channels, skip_block, preprocess):
    """@private
    Producer-side block preparation: load + (optional) mask/skip check + preprocess.

    Returns the `_SKIP` sentinel if the block should be skipped, otherwise a tuple
    `(cpu_tensor, mask_block, inner_bb)` where `cpu_tensor` has a leading batch axis.
    """
    offset = [beg for beg in block.begin]
    inner_bb = tuple(slice(ha, ha + bs) for ha, bs in zip(halo, block.shape))

    mask_block = None
    if mask is not None:
        mask_block, _ = _load_block(mask, offset, block_shape, halo, with_channels=False)
        mask_block = mask_block[inner_bb].astype("bool")
        if mask_block.sum() == 0:
            return _SKIP

    inp, _ = _load_block(input_, offset, block_shape, halo, with_channels=with_channels)

    if skip_block is not None and skip_block(inp):
        return _SKIP

    if preprocess is not None:
        inp = preprocess(inp)

    # add (channel) and batch axis -> [1, (C,) *spatial]
    expand_dims = np.s_[None] if with_channels else np.s_[None, None]
    tensor = torch.from_numpy(inp[expand_dims])
    return tensor, mask_block, inner_bb


def _write_prediction(prediction, block, output, ndim, mask_block, inner_bb, postprocess):
    """@private
    Writer-side logic: postprocess + inner crop + mask-zero + write to `output`."""
    if postprocess is not None:
        prediction = postprocess(prediction)

    if prediction.ndim == ndim + 1:
        inner_bb_pred = (slice(None),) + inner_bb
    else:
        inner_bb_pred = inner_bb
    prediction = prediction[inner_bb_pred]

    if mask_block is not None:
        if prediction.ndim == ndim + 1:
            mb = np.broadcast_to(mask_block[None], prediction.shape)
        else:
            mb = mask_block
        prediction[~mb] = 0

    bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
    if isinstance(output, list):  # multiple outputs: split the prediction channels
        for out, channel_slice in output:
            this_bb = bb if out.ndim == ndim else (slice(None),) + bb
            out[this_bb] = prediction[channel_slice]
    else:  # single output array
        if output.ndim == ndim + 1:
            bb = (slice(None),) + bb
        output[bb] = prediction


def _concurrent_write_safe(arr, block_shape, start):
    """@private
    Whether `arr` can be written to from multiple threads concurrently for the given block grid.

    - numpy arrays are always safe (writers touch disjoint in-memory regions).
    - hdf5 datasets are never safe (h5py is not thread-safe for concurrent writes).
    - zarr / n5 datasets are safe iff the block grid is aligned with the chunks (and the shards,
      for zarr v3), so that each chunk/shard is written by exactly one block.
    - unknown / unchunked backends are treated conservatively as unsafe.
    """
    if isinstance(arr, np.ndarray):
        return True

    module = type(arr).__module__
    if module.startswith("h5py"):
        return False

    chunks = getattr(arr, "chunks", None)
    if chunks is None:  # unknown backend or unchunked -> be conservative
        return False

    # zarr v3 exposes the shard shape via .shards (None if not sharded); the shard is the atomic
    # write unit when present, otherwise the chunk is. getattr covers z5py / older zarr (no shards).
    shards = getattr(arr, "shards", None)
    unit = shards if shards is not None else chunks

    # compare only the spatial axes: every block writes the full channel range at a disjoint
    # spatial bounding box, so channel-axis chunking never causes a write conflict.
    ndim = len(block_shape)
    unit_spatial = tuple(unit[-ndim:])
    if any(bs % u != 0 for bs, u in zip(block_shape, unit_spatial)):
        return False
    if any(s % u != 0 for s, u in zip(start, unit_spatial)):
        return False
    return True


def predict_with_halo_pipelined(
    input_: ArrayLike,
    model: torch.nn.Module,
    gpu_ids: List[Union[str, int]],
    block_shape: Tuple[int, ...],
    halo: Tuple[int, ...],
    output: Optional[Union[ArrayLike, List[Tuple[ArrayLike, slice]]]] = None,
    preprocess: Callable[[Union[torch.Tensor, np.ndarray]], Union[torch.Tensor, np.ndarray]] = standardize,
    postprocess: Callable[[np.ndarray], np.ndarray] = None,
    with_channels: bool = False,
    skip_block: Callable[[Any], bool] = None,
    mask: Optional[ArrayLike] = None,
    disable_tqdm: bool = False,
    tqdm_desc: str = "predict with halo (pipelined)",
    prediction_function: Optional[Callable] = None,
    roi: Optional[Tuple[slice]] = None,
    iter_list: Optional[List[int]] = None,
    batch_size: int = 1,
    num_prefetch_workers: int = 4,
    queue_size: Optional[int] = None,
    num_write_workers: int = 1,
    write_queue_size: Optional[int] = None,
    grid_shift: Optional[Tuple[float, ...]] = None,
) -> ArrayLike:
    """Run block-wise network prediction with a halo, pipelined for higher GPU throughput.

    This is an alternate implementation of `predict_with_halo` that decouples block
    loading, GPU prediction and output writing into a producer-consumer pipeline
    connected by queues:

        producers (CPU threads: load + preprocess) -> input queue
          -> consumer(s), one per GPU (stack a batch, predict, unstack) -> output queue
          -> writer thread(s) (postprocess + write).

    While the GPU works on one batch, the prefetch workers load and preprocess the
    next blocks and the writer drains finished predictions, keeping the GPU fed.
    Blocks can additionally be stacked into batches for one forward pass via `batch_size`.

    The pipeline is thread-based (not multiprocessing) so that lazy hdf5/zarr/n5 inputs
    (whose file handles are not fork/pickle-safe) work, and so that writers can share the
    output array directly. Note that heavy *Python-level* `preprocess`/`postprocess`
    callbacks will not parallelize across prefetch workers due to the GIL; the default
    `standardize` is numpy-vectorized and releases the GIL.

    Args:
        input_: The input data, can be a numpy array, a hdf5/zarr/z5py dataset or similar.
        model: The network.
        gpu_ids: List of device ids to use for prediction. To run prediction on the CPU, pass `["cpu"]`.
            One prediction consumer thread (with its own model replica) is run per device.
        block_shape: The shape of the inner block to use for prediction.
        halo: The shape of the halo to use for prediction.
        output: The output data, will be allocated if None is passed.
            Instead of a single output, this can also be a list of outputs and a slice for the corresponding channel.
        preprocess: Function to preprocess input data before passing it to the network.
        postprocess: Function to postprocess the network predictions.
        with_channels: Whether the input has a channel axis.
        skip_block: Function to evaluate whether a given input block will be skipped.
        mask: Elements outside the mask will be ignored in the prediction.
        disable_tqdm: Flag that allows to disable tqdm output (e.g. if function is called multiple times).
        tqdm_desc: Description shown by the tqdm output.
        prediction_function: A wrapper function for prediction to enable custom prediction procedures.
            It must operate on the leading batch axis; with the default `batch_size=1` it does not need changes.
        roi: A region of interest of the input for which to run prediction.
        iter_list: Optional list of block ids to iterate over.
        batch_size: The number of blocks stacked into a single forward pass. Trades GPU memory for throughput.
        num_prefetch_workers: The number of CPU threads used to load and preprocess blocks.
        queue_size: The maximum size of the input (prefetch) queue. Provides backpressure to bound memory use.
            If None, a value derived from the number of devices and `batch_size` is used.
        num_write_workers: The number of threads used to write predictions. Values > 1 are safe for in-memory
            numpy outputs, and for zarr/n5 outputs whose chunks (and shards, for zarr v3) are aligned with
            block_shape. For hdf5, misaligned zarr/n5, or other outputs this is automatically clamped to 1.
        write_queue_size: The maximum size of the output (write) queue. If None, a default is used.
        grid_shift: Not supported by this function; raises NotImplementedError if passed. Use `predict_with_halo`.

    Returns:
        The model output.
    """
    if grid_shift is not None:
        raise NotImplementedError(
            "grid_shift is not supported by predict_with_halo_pipelined. "
            "Use predict_with_halo for grid_shift, or pre-pad the input and use roi."
        )

    batch_size = max(1, int(batch_size))
    num_prefetch_workers = max(1, int(num_prefetch_workers))
    num_write_workers = max(1, int(num_write_workers))

    devices = [torch.device(gpu) for gpu in gpu_ids]
    models = [
        (model if next(model.parameters()).device == device else deepcopy(model).to(device), device)
        for device in devices
    ]
    n_consumers = len(devices)

    shape0 = input_.shape
    shape_spatial = shape0[1:] if with_channels else shape0
    ndim = len(shape_spatial)
    assert len(block_shape) == len(halo) == ndim

    # blocking
    if roi is None:
        block_start = [0] * ndim
        blocking = bic.utils.Blocking(block_start, list(shape_spatial), block_shape)
    else:
        assert len(roi) == ndim
        block_start = [0 if ro.start is None else ro.start for ro in roi]
        blocking_stop = [sh if ro.stop is None else ro.stop for ro, sh in zip(roi, shape_spatial)]
        blocking = bic.utils.Blocking(block_start, blocking_stop, block_shape)

    # output allocation
    if output is None:
        n_out = models[0][0].out_channels
        output = np.zeros((n_out,) + tuple(shape_spatial), dtype="float32")

    # guard against unsafe concurrent writes: numpy is always safe (disjoint regions),
    # zarr/n5 are safe when their chunks/shards are aligned with the blocks, hdf5 is not.
    if num_write_workers > 1:
        out_arrays = [o for o, _ in output] if isinstance(output, list) else [output]
        if any(not _concurrent_write_safe(o, block_shape, block_start) for o in out_arrays):
            warnings.warn(
                "num_write_workers > 1 requires either an in-memory numpy output or a zarr/n5 "
                "output whose chunks (and shards, for zarr v3) are aligned with block_shape; "
                "falling back to a single writer. HDF5 outputs are never safe for concurrent writes."
            )
            num_write_workers = 1

    # queue sizes
    if queue_size is None:
        queue_size = max(2 * n_consumers * batch_size, 2 * batch_size)
    queue_size = max(queue_size, batch_size)
    if write_queue_size is None:
        write_queue_size = max(2 * n_consumers, 4)

    n_blocks = blocking.number_of_blocks
    iteration_ids = list(range(n_blocks)) if iter_list is None else list(iter_list)
    total = len(iteration_ids)

    # pre-fill the block-id queue with all ids followed by one STOP per producer
    id_queue = queue.Queue()
    for bid in iteration_ids:
        id_queue.put(bid)
    for _ in range(num_prefetch_workers):
        id_queue.put(_STOP)

    input_queue = queue.Queue(maxsize=queue_size)
    output_queue = queue.Queue(maxsize=write_queue_size)

    stop_event = threading.Event()
    error_box = []
    error_lock = threading.Lock()
    progress_lock = threading.Lock()
    pbar = tqdm(total=total, disable=disable_tqdm, desc=tqdm_desc)

    remaining_producers = _AtomicCounter(num_prefetch_workers)
    remaining_consumers = _AtomicCounter(n_consumers)

    def record_error(exc):
        with error_lock:
            if not error_box:
                error_box.append(exc)
        stop_event.set()

    def producer():
        try:
            while True:
                bid = id_queue.get()
                if bid is _STOP or stop_event.is_set():
                    break
                block = blocking.get_block(bid)
                result = _prepare_block_input(
                    input_, mask, block, block_shape, halo, with_channels, skip_block, preprocess
                )
                if result is _SKIP:
                    with progress_lock:
                        pbar.update(1)
                    continue
                tensor, mask_block, inner_bb = result
                _safe_put(input_queue, _BlockJob(block, inner_bb, mask_block, tensor), stop_event)
        except _Aborted:
            pass
        except Exception as e:  # noqa
            record_error(e)
        finally:
            # the last producer to finish signals the consumers (skipped on the abort path,
            # where consumers unwind via _safe_get instead)
            if remaining_producers.decrement() == 0 and not stop_event.is_set():
                for _ in range(n_consumers):
                    input_queue.put(_STOP)

    def consumer(worker_id):
        net, device = models[worker_id]
        try:
            while True:
                jobs = []
                got_stop = False
                while len(jobs) < batch_size:
                    item = _safe_get(input_queue, stop_event)
                    if item is _STOP:
                        got_stop = True
                        break
                    jobs.append(item)

                if jobs:  # run (possibly partial) batch
                    batch = torch.cat([job.tensor for job in jobs], dim=0).to(device)
                    with torch.no_grad():
                        prediction = net(batch) if prediction_function is None \
                            else prediction_function(net, batch)
                    if not torch.is_tensor(prediction):  # list/tuple of outputs -> take the first
                        prediction = prediction[0]
                    prediction = prediction.cpu().numpy()
                    for i, job in enumerate(jobs):
                        job.prediction = np.array(prediction[i])
                        job.tensor = None
                        _safe_put(output_queue, job, stop_event)

                if got_stop:
                    break
        except _Aborted:
            pass
        except Exception as e:  # noqa
            record_error(e)
        finally:
            if remaining_consumers.decrement() == 0 and not stop_event.is_set():
                for _ in range(num_write_workers):
                    output_queue.put(_STOP)

    def writer():
        try:
            while True:
                job = _safe_get(output_queue, stop_event)
                if job is _STOP:
                    break
                _write_prediction(
                    job.prediction, job.block, output, ndim, job.mask_block, job.inner_bb, postprocess
                )
                with progress_lock:
                    pbar.update(1)
        except _Aborted:
            pass
        except Exception as e:  # noqa
            record_error(e)

    writers = [threading.Thread(target=writer, name=f"predict-writer-{i}") for i in range(num_write_workers)]
    consumers = [threading.Thread(target=consumer, args=(i,), name=f"predict-consumer-{i}")
                 for i in range(n_consumers)]
    producers = [threading.Thread(target=producer, name=f"predict-producer-{i}")
                 for i in range(num_prefetch_workers)]
    threads = writers + consumers + producers

    try:
        for t in writers:
            t.start()
        for t in consumers:
            t.start()
        for t in producers:
            t.start()

        for t in producers:
            t.join()
        for t in consumers:
            t.join()
        for t in writers:
            t.join()
    finally:
        stop_event.set()
        for t in threads:
            t.join()
        pbar.close()

    if error_box:
        raise error_box[0]

    return output
