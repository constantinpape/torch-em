import numpy as np
from torch.utils.data import Dataset


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

        # compute the number of samples for each volume
        self.ds_lens = [len(dataset) for dataset in self.datasets]
        self._len = sum(self.ds_lens)

        # compute the offsets for the samples
        self.ds_offsets = np.cumsum(self.ds_lens)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        # find the dataset id corresponding to this index
        ds_idx = 0
        while True:
            if idx < self.ds_offsets[ds_idx]:
                break
            ds_idx += 1

        # get sample from the dataset
        ds = self.datasets[ds_idx]
        offset = self.ds_offsets[ds_idx - 1] if ds_idx > 0 else 0
        idx_in_ds = idx - offset
        assert idx_in_ds < len(ds) and idx_in_ds >= 0, f"Failed with: {idx_in_ds}, {len(ds)}"
        return ds[idx_in_ds]
