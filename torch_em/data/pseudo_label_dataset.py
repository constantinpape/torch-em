import os
from typing import Union, Tuple, Optional, List, Any, Callable

import torch

from .raw_dataset import RawDataset
from ..util import ensure_tensor_with_channels


class PseudoLabelDataset(RawDataset):
    """Dataset that uses a prediction function to provide raw data and pseudo labels for segmentation training.

    The dataset loads a patch from the raw data and then applies the pseudo labeler to it to predict pseudo labels.
    The raw data and pseudo labels are returned together as a sample for a batch.
    The datataset supports all file formats that can be opened with `elf.io.open_file`, such as hdf5, zarr or n5.
    Use `raw_path` to specify the path to the file and `raw_key` to specify the internal dataset.
    It also supports regular image formats, such as .tif. For these cases set `raw_key=None`.

    Args:
        raw_path: The file path to the raw image data. May also be a list of file paths.
        raw_key: The key to the internal dataset containing the raw data.
        patch_shape: The patch shape for a training sample.
        pseudo_labeler: The pseudo labeler. Must be a function that accepts the raw data as torch tensor
            and that returns the predicted labels as torch tensor.
        raw_transform: Transformation applied to the raw data of a sample.
        label_transform: Transformation applied to the label data of a sample.
        roi: Region of interest in the raw data.
            If given, the raw data will only be loaded from the corresponding area.
        dtype: The return data type of the raw data.
        n_samples: The length of this dataset. If None, the length will be set to `len(raw_image_paths)`.
        sampler: Sampler for rejecting samples according to a defined criterion.
            The sampler must be a callable that accepts the raw data and label data (as numpy arrays) as input.
        ndim: The spatial dimensionality of the data. If None, will be derived from the raw data.
        with_channels: Whether the raw data has channels.
        labeler_device: The expected device for the pseudo labeler.
    """
    def __init__(
        self,
        raw_path: Union[List[Any], str, os.PathLike],
        raw_key: Optional[str],
        patch_shape: Tuple[int, ...],
        pseudo_labeler: Callable,
        raw_transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        roi: Optional[Union[slice, Tuple[slice, ...]]] = None,
        dtype: torch.dtype = torch.float32,
        n_samples: Optional[int] = None,
        sampler: Optional[Callable] = None,
        ndim: Optional[Union[int]] = None,
        with_channels: bool = False,
        labeler_device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__(
            raw_path, raw_key, patch_shape, raw_transform=raw_transform, transform=transform, roi=roi,
            dtype=dtype, n_samples=n_samples, sampler=sampler, ndim=ndim, with_channels=with_channels
        )
        self.pseudo_labeler = pseudo_labeler
        self.label_transform = label_transform
        self.labeler_device = next(pseudo_labeler.parameters()).device if labeler_device is None else labeler_device

    def __getitem__(self, index):
        raw = self._get_sample(index)

        # Transform for augmentations.
        # Applied to the raw data since, labels are generated on the fly by the pseudo_labeler.
        if self.transform is not None:
            raw = self.transform(raw)[0]
            if self.trafo_halo is not None:
                raw = self.crop(raw)

        raw = ensure_tensor_with_channels(raw, ndim=self._ndim, dtype=self.dtype)
        with torch.no_grad():
            # Ilastik needs uint as input, so normalize afterwards.
            labels = self.pseudo_labeler(raw[None].to(self.labeler_device))[0]

        # Normalize the raw data.
        if self.raw_transform is not None:
            raw = self.raw_transform(raw.cpu().detach().numpy())
        raw = ensure_tensor_with_channels(raw, ndim=self._ndim, dtype=self.dtype)

        if self.label_transform is not None:
            labels = self.label_transform(labels)
        labels = ensure_tensor_with_channels(labels, ndim=self._ndim)

        return raw, labels
