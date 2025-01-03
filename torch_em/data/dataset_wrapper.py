from typing import Callable
from collections.abc import Sized

from torch.utils.data import Dataset


class DatasetWrapper(Dataset):
    """Wrapper around a dataset that applies a function to items before retrieval.

    Args:
        dataset: The datast.
        wrap_item: The function to apply to items before retrieval.
    """
    def __init__(self, dataset: Dataset, wrap_item: Callable):
        assert isinstance(dataset, Dataset) and isinstance(dataset, Sized), "iterable datasets not supported"
        self.dataset = dataset
        self.wrap_item = wrap_item

    def __getitem__(self, item):
        return self.wrap_item(self.dataset[item])

    def __len__(self):
        return len(self.dataset)
