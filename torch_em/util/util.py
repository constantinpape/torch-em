import numpy as np
import torch

# torch doesn't support most unsigned types,
# so we map them to their signed equivalent
DTYPE_MAP = {
    np.dtype('uint16'): np.int16,
    np.dtype('uint32'): np.int32,
    np.dtype('uint64'): np.int64
}


def ensure_tensor(tensor, dtype=None):

    if isinstance(tensor, np.ndarray):
        if np.dtype(tensor.dtype) in DTYPE_MAP:
            tensor = tensor.astype(DTYPE_MAP[tensor.dtype])
        tensor = torch.from_numpy(tensor)

    assert torch.is_tensor(tensor), f"Cannot convert {type(tensor)} to torch"
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def ensure_tensor_with_channels(tensor, ndim, dtype=None):
    assert ndim in (2, 3)
    tensor = ensure_tensor(tensor, dtype)
    if ndim == 2:
        assert tensor.ndim in (2, 3, 4, 5), str(tensor.ndim)
        if tensor.ndim == 2:
            tensor = tensor[None]
        elif tensor.ndim == 4:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        elif tensor.ndim == 5:
            assert tensor.shape[:2] == (1, 1)
            tensor = tensor[0, 0]
    else:
        assert tensor.ndim in (3, 4, 5)
        if tensor.ndim == 3:
            tensor = tensor[None]
        elif tensor.ndim == 5:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
    return tensor


def ensure_array(array, dtype=None):
    if torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    assert isinstance(array, np.ndarray), f"Cannot convert {type(array)} to numpy"
    if dtype is not None:
        array = np.require(array, dtype=dtype)
    return array


def ensure_spatial_array(array, ndim, dtype=None):
    assert ndim in (2, 3)
    array = ensure_array(array, dtype)
    if ndim == 2:
        assert array.ndim in (2, 3, 4, 5), str(array.ndim)
        if array.ndim == 3:
            assert array.shape[0] == 1
            array = array[0]
        elif array.ndim == 4:
            assert array.shape[:2] == (1, 1)
            array = array[0, 0]
        elif array.ndim == 5:
            assert array.shape[:3] == (1, 1, 1)
            array = array[0, 0, 0]
    else:
        assert array.ndim in (3, 4, 5), str(array.ndim)
        if array.ndim == 4:
            assert array.shape[0] == 1
            array = array[0]
        elif array.ndim == 5:
            assert array.shape[:2] == (1, 1)
            array = array[0, 0]
    return array
