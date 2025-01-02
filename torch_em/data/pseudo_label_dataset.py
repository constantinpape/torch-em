import os
from typing import Union, Tuple, Optional, List, Any, Callable

import torch

from .raw_dataset import RawDataset
from ..util import ensure_tensor_with_channels


class PseudoLabelDataset(RawDataset):
    def __init__(
        self,
        raw_path: Union[List[Any], str, os.PathLike],
        raw_key: Optional[str],
        patch_shape: Tuple[int, ...],
        pseudo_labeler: Callable,
        raw_transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        roi: Optional[dict] = None,
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

        # transform for augmentations
        # only applied to raw since labels are generated on the fly anyway by the pseudo_labeler
        if self.transform is not None:
            raw = self.transform(raw)[0]
            if self.trafo_halo is not None:
                raw = self.crop(raw)

        raw = ensure_tensor_with_channels(raw, ndim=self._ndim, dtype=self.dtype)
        with torch.no_grad():
            labels = self.pseudo_labeler(
                raw[None].to(self.labeler_device))[0]  # ilastik needs uint input, so normalize afterwards

        # normalize after ilastik
        if self.raw_transform is not None:
            raw = self.raw_transform(
                raw.cpu().detach().numpy()
            )  # normalization functions need numpy array, self.transform already creates torch.tensor

        raw = ensure_tensor_with_channels(raw, ndim=self._ndim, dtype=self.dtype)

        if self.label_transform is not None:
            labels = self.label_transform(labels)
        labels = ensure_tensor_with_channels(labels, ndim=self._ndim)

        return raw, labels
