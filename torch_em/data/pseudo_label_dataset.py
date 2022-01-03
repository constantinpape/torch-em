import torch
from .raw_dataset import RawDataset
from ..util import ensure_tensor_with_channels


class PseudoLabelDataset(RawDataset):
    def __init__(
        self,
        raw_path,
        raw_key,
        patch_shape,
        pseudo_labeler,
        raw_transform=None,
        label_transform=None,
        transform=None,
        roi=None,
        dtype=torch.float32,
        n_samples=None,
        sampler=None,
        ndim=None,
        with_channels=False,
        labeler_device=None,
    ):
        super().__init__(raw_path, raw_key, patch_shape, raw_transform=raw_transform, transform=transform,
                         roi=roi, dtype=dtype, n_samples=n_samples, sampler=sampler,
                         ndim=ndim, with_channels=with_channels)
        self.pseudo_labeler = pseudo_labeler
        self.label_transform = label_transform
        self.labeler_device = next(pseudo_labeler.parameters()).device if labeler_device is None else labeler_device

    def __getitem__(self, index):
        raw = self._get_sample(index)

        if self.raw_transform is not None:
            raw = self.raw_transform(raw)

        if self.transform is not None:
            raw = self.transform(raw)[0]
            if self.trafo_halo is not None:
                raw = self.crop(raw)

        raw = ensure_tensor_with_channels(raw, ndim=self._ndim, dtype=self.dtype)
        with torch.no_grad():
            labels = self.pseudo_labeler(raw[None].to(self.labeler_device))[0]
        if self.label_transform is not None:
            labels = self.label_transform(labels)
        labels = ensure_tensor_with_channels(labels, ndim=self._ndim)

        return raw, labels
