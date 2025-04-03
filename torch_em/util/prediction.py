from copy import deepcopy
from concurrent import futures
from typing import Tuple, Union, Callable, Any, List, Optional

import numpy as np
import nifty.tools as nt
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

    Returns:
        The model output.
    """
    devices = [torch.device(gpu) for gpu in gpu_ids]
    models = [
        (model if next(model.parameters()).device == device else deepcopy(model).to(device), device)
        for device in devices
    ]

    n_workers = len(gpu_ids)
    shape = input_.shape
    if with_channels:
        shape = shape[1:]

    ndim = len(shape)
    assert len(block_shape) == len(halo) == ndim

    if roi is None:
        blocking = nt.blocking([0] * ndim, shape, block_shape)
    else:
        assert len(roi) == ndim
        blocking_start = [0 if ro.start is None else ro.start for ro in roi]
        blocking_stop = [sh if ro.stop is None else ro.stop for ro, sh in zip(roi, shape)]
        blocking = nt.blocking(blocking_start, blocking_stop, block_shape)

    if output is None:
        n_out = models[0][0].out_channels
        output = np.zeros((n_out,) + shape, dtype="float32")

    def predict_block(block_id):
        worker_id = block_id % n_workers
        net, device = models[worker_id]

        with torch.no_grad():
            block = blocking.getBlock(block_id)
            offset = [beg for beg in block.begin]
            inner_bb = tuple(slice(ha, ha + bs) for ha, bs in zip(halo, block.shape))

            if mask is not None:
                mask_block, _ = _load_block(mask, offset, block_shape, halo, with_channels=False)
                mask_block = mask_block[inner_bb].astype("bool")
                if mask_block.sum() == 0:
                    return

            inp, _ = _load_block(input_, offset, block_shape, halo, with_channels=with_channels)

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
                inner_bb = (slice(None),) + inner_bb
            prediction = prediction[inner_bb]

            if mask is not None:
                if prediction.ndim == ndim + 1:
                    mask_block = np.concatenate(prediction.shape[0] * [mask_block[None]], axis=0)
                prediction[~mask_block] = 0

            bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
            if isinstance(output, list):  # we have multiple outputs and split the prediction channels
                for out, channel_slice in output:
                    this_bb = bb if out.ndim == ndim else (slice(None),) + bb
                    out[this_bb] = prediction[channel_slice]

            else:  # we only have a single output array
                if output.ndim == ndim + 1:
                    bb = (slice(None),) + bb
                output[bb] = prediction

    n_blocks = blocking.numberOfBlocks
    if iter_list is None:
        iteration_ids = range(n_blocks)
    else:
        iteration_ids = np.array(iter_list)
    with futures.ThreadPoolExecutor(n_workers) as tp:
        list(tqdm(tp.map(predict_block, iteration_ids), total=n_blocks, disable=disable_tqdm, desc=tqdm_desc))

    return output
