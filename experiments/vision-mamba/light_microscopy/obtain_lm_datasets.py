import os

import numpy as np
from math import ceil, floor

import torch

import torch_em
import torch_em.data.datasets as datasets
from torch_em.data import MinInstanceSampler, ConcatDataset
from torch_em.transform.label import PerObjectDistanceTransform
from torch_em.transform.raw import normalize_percentile, standardize


class ResizeRawTrafo:
    def __init__(self, desired_shape, do_rescaling=True, padding="constant"):
        self.desired_shape = desired_shape
        self.padding = padding
        self.do_rescaling = do_rescaling

    def __call__(self, raw):
        if self.do_rescaling:
            raw = normalize_percentile(raw, axis=(1, 2))
            raw = np.mean(raw, axis=0)
            raw = standardize(raw)

        tmp_ddim = (self.desired_shape[0] - raw.shape[0], self.desired_shape[1] - raw.shape[1])
        ddim = (tmp_ddim[0] / 2, tmp_ddim[1] / 2)
        raw = np.pad(
            raw,
            pad_width=((ceil(ddim[0]), floor(ddim[0])), (ceil(ddim[1]), floor(ddim[1]))),
            mode=self.padding
        )
        assert raw.shape == self.desired_shape
        return raw


class ResizeLabelTrafo:
    def __init__(self, desired_shape, padding="constant", min_size=0):
        self.desired_shape = desired_shape
        self.padding = padding
        self.min_size = min_size

    def __call__(self, labels):
        distance_trafo = PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False,
            foreground=True, instances=False, min_size=self.min_size
        )
        labels = distance_trafo(labels)

        # choosing H and W from labels (4, H, W), from above dist trafo outputs
        tmp_ddim = (self.desired_shape[0] - labels.shape[1], self.desired_shape[0] - labels.shape[2])
        ddim = (tmp_ddim[0] / 2, tmp_ddim[1] / 2)
        labels = np.pad(
            labels,
            pad_width=((0, 0), (ceil(ddim[0]), floor(ddim[0])), (ceil(ddim[1]), floor(ddim[1]))),
            mode=self.padding
        )
        assert labels.shape[1:] == self.desired_shape, labels.shape
        return labels


def neurips_raw_trafo(raw):
    raw = datasets.neurips_cell_seg.to_rgb(raw)  # ensures 3 channels for the neurips data
    raw = normalize_percentile(raw)
    raw = np.mean(raw, axis=0)
    raw = standardize(raw)
    return raw


def get_concat_lm_datasets(input_path, patch_shape, split_choice):
    assert split_choice in ["train", "val"]

    label_dtype = torch.float32
    sampler = MinInstanceSampler()

    def get_label_transform(min_size=0):
        label_transform = PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False,
            foreground=True, instances=False, min_size=min_size
        )
        return label_transform

    def get_ctc_datasets(
        input_path, patch_shape, sampler, label_transform,
        ignore_datasets=["Fluo-N2DH-GOWT1", "Fluo-N2DL-HeLa"]
    ):
        all_ctc_datasets = []
        for dataset_name in datasets.ctc.CTC_CHECKSUMS["train"].keys():
            if dataset_name in ignore_datasets:
                continue

            all_ctc_datasets.append(
                datasets.get_ctc_segmentation_dataset(
                    path=os.path.join(input_path, "ctc"), dataset_name=dataset_name, patch_shape=(1, *patch_shape),
                    sampler=sampler, label_transform=label_transform, download=True
                )
            )
        return all_ctc_datasets

    _datasets = [
        datasets.get_tissuenet_dataset(
            path=os.path.join(input_path, "tissuenet"), split=split_choice, download=True, patch_shape=patch_shape,
            raw_channel="rgb", label_channel="cell", sampler=sampler, label_dtype=label_dtype,
            raw_transform=ResizeRawTrafo(patch_shape), label_transform=ResizeLabelTrafo(patch_shape, min_size=0),
            n_samples=1000 if split_choice == "train" else 100
        ),
        datasets.get_livecell_dataset(
            path=os.path.join(input_path, "livecell"), split=split_choice, patch_shape=patch_shape,
            download=True, label_transform=get_label_transform(), sampler=sampler, label_dtype=label_dtype,
        ),
        datasets.get_deepbacs_dataset(
            path=os.path.join(input_path, "deepbacs"), split=split_choice, patch_shape=patch_shape,
            label_transform=get_label_transform(), label_dtype=label_dtype,
            download=True, sampler=MinInstanceSampler(min_num_instances=4)
        ),
        datasets.get_neurips_cellseg_supervised_dataset(
            root=os.path.join(input_path, "neurips-cell-seg", "zenodo"), split=split_choice,
            patch_shape=patch_shape, raw_transform=neurips_raw_trafo, label_transform=get_label_transform(),
            label_dtype=label_dtype, sampler=MinInstanceSampler(min_num_instances=3)
        ),
        datasets.get_dsb_dataset(
            path=os.path.join(input_path, "dsb"), split=split_choice if split_choice == "train" else "test",
            patch_shape=patch_shape, label_transform=get_label_transform(), sampler=sampler,
            label_dtype=label_dtype, download=True,
        ),
        datasets.get_plantseg_dataset(
            path=os.path.join(input_path, "plantseg"), name="root", sampler=MinInstanceSampler(min_num_instances=10),
            patch_shape=(1, *patch_shape), download=True, split=split_choice, ndim=2, label_dtype=label_dtype,
            raw_transform=ResizeRawTrafo(patch_shape, do_rescaling=False),
            label_transform=ResizeLabelTrafo(patch_shape, min_size=0),
            n_samples=1000 if split_choice == "train" else 100
        ),
    ]
    if split_choice == "train":
        _datasets += get_ctc_datasets(input_path, patch_shape, sampler, label_transform=get_label_transform())

    dataset = ConcatDataset(*_datasets)

    # increasing the sampling attempts for the neurips cellseg dataset
    dataset.datasets[3].max_sampling_attempts = 5000

    return dataset


def get_lm_loaders(input_path, patch_shape):
    """This returns the concatenated light microscopy datasets implemented in torch_em:
    https://github.com/constantinpape/torch-em/tree/main/torch_em/data/datasets
    It will automatically download all the datasets.
    """
    train_dataset = get_concat_lm_datasets(input_path, patch_shape, "train")
    val_dataset = get_concat_lm_datasets(input_path, patch_shape, "val")
    train_loader = torch_em.get_data_loader(train_dataset, batch_size=8, shuffle=True, num_workers=16)
    val_loader = torch_em.get_data_loader(val_dataset, batch_size=1, shuffle=True, num_workers=16)
    return train_loader, val_loader
