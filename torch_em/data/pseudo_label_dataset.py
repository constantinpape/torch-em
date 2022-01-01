import torch
from .raw_dataset import RawDataset
from ..util import ensure_tensor_with_channels


class PseudoLabelDataset(RawDataset):
    def __init__(
        self,
        raw_path,
        raw_key,
        patch_shape,
        pseudo_labler,
        raw_transform=None,
        label_transform=None,
        transform=None,
        roi=None,
        dtype=torch.float32,
        n_samples=None,
        sampler=None,
        ndim=None,
        with_channels=False,
    ):
        super().__init__(raw_path, raw_key, patch_shape, raw_transform=raw_transform, transform=transform,
                         roi=roi, dtype=dtype, n_samples=n_samples, sampler=sampler,
                         ndim=ndim, with_channels=with_channels)
        self.pseudo_labler = pseudo_labler
        self.label_transform = label_transform

    def __getitem__(self, index):
        raw = self._get_sample(index)

        if self.raw_transform is not None:
            raw = self.raw_transform(raw)

        if self.transform is not None:
            raw = self.transform(raw)
            if self.trafo_halo is not None:
                raw = self.crop(raw)

        raw = ensure_tensor_with_channels(raw, ndim=self._ndim, dtype=self.dtype)
        labels = self.pseudo_labler(raw)
        if self.label_transform is not None:
            labels = self.label_transform(labels)

        return raw, labels
