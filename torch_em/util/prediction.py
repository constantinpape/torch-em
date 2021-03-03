from concurrent import futures
from copy import deepcopy

import nifty.tools as nt
import numpy as np
import torch
from tqdm import tqdm


def _load_block(input_, offset, block_shape, halo,
                padding_mode='reflect'):
    shape = input_.shape

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
    data = input_[bb]

    ndim = len(shape)
    # pad if necessary
    if pad_left is not None or pad_right is not None:
        pad_left = (0,) * ndim if pad_left is None else pad_left
        pad_right = (0,) * ndim if pad_right is None else pad_right
        pad_width = tuple((pl, pr) for pl, pr in zip(pad_left, pad_right))
        data = np.pad(data, pad_width, mode=padding_mode)

        # extend the bounding box for downstream
        bb = tuple(
            slice(b.start - pl, b.stop + pr)
            for b, pl, pr in zip(bb, pad_left, pad_right)
        )

    return data, bb


# TODO support input channels
def predict_with_halo(
    input_,
    model,
    gpu_ids,
    block_shape,
    halo,
    output=None,
    preprocess=None,
    postprocess=None
):
    """ Run block-wise network prediction with halo.

    Arguments:
        input_ [arraylike] - the input data, can be a numpy array, a hdf5/zarr/z5py dataset or similar
        model [nn.Module] - the network
        gpu_ids [list[int or string]] - list of gpus id used for prediction
        block_shape [tuple] - shape of inner block used for prediction
        halo [tuple] - shape of halo used for prediction
        output [arraylike] - output data, will be allocated if None (default: None)
        preprocess [callable] - function to preprocess input data before passing it to the network (default: None)
        postprocess [callable] - function to postprocess the network predictions (default: None)
    """
    devices = [torch.device(gpu) for gpu in gpu_ids]
    models = [
        (model if next(model.parameters()).device == device else deepcopy(model).to(device), device)
        for device in devices
    ]

    n_workers = len(gpu_ids)
    shape = input_.shape
    ndim = len(shape)
    blocking = nt.blocking([0] * len(shape), shape, block_shape)

    if output is None:
        n_out = models[0][0].out_channels
        output = np.zeros((n_out,) + shape, dtype='float32')

    def predict_block(block_id):
        worker_id = block_id % n_workers
        net, device = models[worker_id]

        with torch.no_grad():
            block = blocking.getBlock(block_id)
            offset = [beg for beg in block.begin]
            inp, _ = _load_block(input_, offset, block_shape, halo)
            if preprocess is not None:
                inp = preprocess(inp)

            inp = torch.from_numpy(inp[None, None]).to(device)

            out = net(inp)
            # allow for list of tensors
            try:
                out = out.cpu().numpy().squeeze(0)
            except AttributeError:
                out = out[0]
                out = out.cpu().numpy().squeeze(0)

            if postprocess is not None:
                out = postprocess(out)

            inner_bb = tuple(slice(ha, ha + bs) for ha, bs in zip(halo, block.shape))
            if out.ndim == ndim + 1:
                inner_bb = (slice(None),) + inner_bb
            out = out[inner_bb]

            bb_out = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
            if output.ndim == ndim + 1:
                bb_out = (slice(None),) + bb_out
            output[bb_out] = out

    n_blocks = blocking.numberOfBlocks
    with futures.ThreadPoolExecutor(n_workers) as tp:
        list(tqdm(tp.map(predict_block, range(n_blocks)), total=n_blocks))

    return output
