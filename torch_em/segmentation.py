import os
from glob import glob
from typing import Any, Dict, Optional, Union, Tuple, List, Callable

import torch
import torch.utils.data
from torch.utils.data import DataLoader

from .loss import DiceLoss
from .util import load_data
from .trainer import DefaultTrainer
from .trainer.tensorboard_logger import TensorboardLogger
from .transform import get_augmentations, get_raw_transform
from .data import ConcatDataset, ImageCollectionDataset, SegmentationDataset


# TODO add a heuristic to estimate this from the number of epochs
DEFAULT_SCHEDULER_KWARGS = {"mode": "min", "factor": 0.5, "patience": 5}
"""@private
"""


#
# convenience functions for segmentation loaders
#

# TODO implement balanced and make it the default
# def samples_to_datasets(n_samples, raw_paths, raw_key, split="balanced"):
def samples_to_datasets(n_samples, raw_paths, raw_key, split="uniform"):
    """@private
    """
    assert split in ("balanced", "uniform")
    n_datasets = len(raw_paths)
    if split == "uniform":
        # even distribution of samples to datasets
        samples_per_ds = n_samples // n_datasets
        divider = n_samples % n_datasets
        return [samples_per_ds + 1 if ii < divider else samples_per_ds for ii in range(n_datasets)]
    else:
        # distribution of samples to dataset based on the dataset lens
        raise NotImplementedError


def check_paths(raw_paths, label_paths):
    """@private
    """
    if not isinstance(raw_paths, type(label_paths)):
        raise ValueError(f"Expect raw and label paths of same type, got {type(raw_paths)}, {type(label_paths)}")

    def _check_path(path):
        if isinstance(path, str):
            if not os.path.exists(path):
                raise ValueError(f"Could not find path {path}")
        else:
            # check for single path or multiple paths (for same volume - supports multi-modal inputs)
            for per_path in path:
                if not os.path.exists(per_path):
                    raise ValueError(f"Could not find path {per_path}")

    if isinstance(raw_paths, str):
        _check_path(raw_paths)
        _check_path(label_paths)
    else:
        if len(raw_paths) != len(label_paths):
            raise ValueError(f"Expect same number of raw and label paths, got {len(raw_paths)}, {len(label_paths)}")
        for rp, lp in zip(raw_paths, label_paths):
            _check_path(rp)
            _check_path(lp)


# Check if we can load the data as SegmentationDataset.
def is_segmentation_dataset(raw_paths, raw_key, label_paths, label_key):
    """@private
    """
    def _can_open(path, key):
        try:
            load_data(path, key)
            return True
        except Exception:
            return False

    if isinstance(raw_paths, str):
        can_open_raw = _can_open(raw_paths, raw_key)
        can_open_label = _can_open(label_paths, label_key)
    else:
        can_open_raw = [_can_open(rp, raw_key) for rp in raw_paths]
        if not can_open_raw.count(can_open_raw[0]) == len(can_open_raw):
            raise ValueError("Inconsistent raw data")
        can_open_raw = can_open_raw[0]

        can_open_label = [_can_open(lp, label_key) for lp in label_paths]
        if not can_open_label.count(can_open_label[0]) == len(can_open_label):
            raise ValueError("Inconsistent label data")
        can_open_label = can_open_label[0]

    if can_open_raw != can_open_label:
        raise ValueError("Inconsistent raw and label data")

    return can_open_raw


def _load_segmentation_dataset(raw_paths, raw_key, label_paths, label_key, **kwargs):
    rois = kwargs.pop("rois", None)
    if isinstance(raw_paths, str):
        if rois is not None:
            assert isinstance(rois, (tuple, slice))
            if isinstance(rois, tuple):
                assert all(isinstance(roi, slice) for roi in rois)
        ds = SegmentationDataset(raw_paths, raw_key, label_paths, label_key, roi=rois, **kwargs)
    else:
        assert len(raw_paths) > 0
        if rois is not None:
            assert len(rois) == len(label_paths)
            assert all(isinstance(roi, tuple) for roi in rois), f"{rois}"
        n_samples = kwargs.pop("n_samples", None)

        samples_per_ds = (
            [None] * len(raw_paths) if n_samples is None else samples_to_datasets(n_samples, raw_paths, raw_key)
        )
        ds = []
        for i, (raw_path, label_path) in enumerate(zip(raw_paths, label_paths)):
            roi = None if rois is None else rois[i]
            dset = SegmentationDataset(
                raw_path, raw_key, label_path, label_key, roi=roi, n_samples=samples_per_ds[i], **kwargs
            )
            ds.append(dset)
        ds = ConcatDataset(*ds)
    return ds


def _load_image_collection_dataset(raw_paths, raw_key, label_paths, label_key, roi, **kwargs):
    def _get_paths(rpath, rkey, lpath, lkey, this_roi):
        rpath = glob(os.path.join(rpath, rkey))
        rpath.sort()
        if len(rpath) == 0:
            raise ValueError(f"Could not find any images for pattern {os.path.join(rpath, rkey)}")

        lpath = glob(os.path.join(lpath, lkey))
        lpath.sort()
        if len(rpath) != len(lpath):
            raise ValueError(f"Expect same number of raw and label images, got {len(rpath)}, {len(lpath)}")

        if this_roi is not None:
            rpath, lpath = rpath[roi], lpath[roi]

        return rpath, lpath

    patch_shape = kwargs.pop("patch_shape")
    if patch_shape is not None:
        if len(patch_shape) == 3:
            if patch_shape[0] != 1:
                raise ValueError(f"Image collection dataset expects 2d patch shape, got {patch_shape}")
            patch_shape = patch_shape[1:]
        assert len(patch_shape) == 2

    if isinstance(raw_paths, str):
        raw_paths, label_paths = _get_paths(raw_paths, raw_key, label_paths, label_key, roi)
        ds = ImageCollectionDataset(raw_paths, label_paths, patch_shape=patch_shape, **kwargs)

    elif raw_key is None:
        assert label_key is None
        assert isinstance(raw_paths, (list, tuple)) and isinstance(label_paths, (list, tuple))
        assert len(raw_paths) == len(label_paths)
        ds = ImageCollectionDataset(raw_paths, label_paths, patch_shape=patch_shape, **kwargs)

    else:
        ds = []
        n_samples = kwargs.pop("n_samples", None)
        samples_per_ds = (
            [None] * len(raw_paths) if n_samples is None else samples_to_datasets(n_samples, raw_paths, raw_key)
        )
        if roi is None:
            roi = len(raw_paths) * [None]
        assert len(roi) == len(raw_paths)
        for i, (raw_path, label_path, this_roi) in enumerate(zip(raw_paths, label_paths, roi)):
            print(raw_path, label_path, this_roi)
            rpath, lpath = _get_paths(raw_path, raw_key, label_path, label_key, this_roi)
            dset = ImageCollectionDataset(rpath, lpath, patch_shape=patch_shape, n_samples=samples_per_ds[i], **kwargs)
            ds.append(dset)
        ds = ConcatDataset(*ds)

    return ds


def _get_default_transform(path, key, is_seg_dataset, ndim):
    if is_seg_dataset and ndim is None:
        shape = load_data(path, key).shape
        if len(shape) == 2:
            ndim = 2
        else:
            # heuristics to figure out whether to use default 3d
            # or default anisotropic augmentations
            ndim = "anisotropic" if shape[0] < shape[1] // 2 else 3

    elif is_seg_dataset and ndim is not None:
        pass

    else:
        ndim = 2

    return get_augmentations(ndim)


def default_segmentation_loader(
    raw_paths: Union[List[Any], str, os.PathLike],
    raw_key: Optional[str],
    label_paths: Union[List[Any], str, os.PathLike],
    label_key: Optional[str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    label_transform: Optional[Callable] = None,
    label_transform2: Optional[Callable] = None,
    raw_transform: Optional[Callable] = None,
    transform: Optional[Callable] = None,
    dtype: torch.device = torch.float32,
    label_dtype: torch.device = torch.float32,
    rois: Optional[Union[slice, Tuple[slice, ...]]] = None,
    n_samples: Optional[int] = None,
    sampler: Optional[Callable] = None,
    ndim: Optional[int] = None,
    is_seg_dataset: Optional[bool] = None,
    with_channels: bool = False,
    with_label_channels: bool = False,
    verify_paths: bool = True,
    with_padding: bool = True,
    z_ext: Optional[int] = None,
    **loader_kwargs,
) -> torch.utils.data.DataLoader:
    """Get data loader for training a segmentation network.

    See `torch_em.data.SegmentationDataset` and `torch_em.data.ImageCollectionDataset` for details
    on the data formats that are supported.

    Args:
        raw_paths: The file path(s) to the raw data. Can either be a single path or multiple file paths.
        raw_key: The name of the internal dataset containing the raw data. Set to None for regular image files.
        label_paths: The file path(s) to the label data. Can either be a single path or multiple file paths.
        label_key: The name of the internal dataset containing the raw data. Set to None for regular image files.
        batch_size: The batch size for the data loader.
        patch_shape: The patch shape for the training samples.
        label_transform: Transformation applied to the label data of a sample,
            before applying augmentations via `transform`.
        label_transform2: Transformation applied to the label data of a sample,
            after applying augmentations via `transform`.
        raw_transform: Transformation applied to the raw data of a sample,
            before applying augmentations via `transform`.
        transform: Transformation applied to both the raw data and label data of a sample.
            This can be used to implement data augmentations.
        dtype: The return data type of the raw data.
        label_dtype: The return data type of the label data.
        rois: Regions of interest in the data.  If given, the data will only be loaded from the corresponding area.
        n_samples: The length of the underlying dataset. If None, the length will be set to `len(raw_paths)`.
        sampler: Sampler for rejecting samples according to a defined criterion.
            The sampler must be a callable that accepts the raw data (as numpy arrays) as input.
        ndim: The spatial dimensionality of the data. If None, will be derived from the raw data.
        is_seg_dataset: Whether this is a segmentation dataset or an image collection dataset.
            If None, the type of dataset will be derived from the data.
        with_channels: Whether the raw data has channels.
        with_label_channels: Whether the label data has channels.
        verify_paths: Whether to verify all paths before creating the dataset.
        with_padding: Whether to pad samples to `patch_shape` if their shape is smaller.
        z_ext: Extra bounding box for loading the data across z.
        loader_kwargs: Keyword arguments for `torch.utils.data.DataLoder`.

    Returns:
        The torch data loader.
    """
    ds = default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=raw_key,
        label_paths=label_paths,
        label_key=label_key,
        patch_shape=patch_shape,
        label_transform=label_transform,
        label_transform2=label_transform2,
        raw_transform=raw_transform,
        transform=transform,
        dtype=dtype,
        label_dtype=label_dtype,
        rois=rois,
        n_samples=n_samples,
        sampler=sampler,
        ndim=ndim,
        is_seg_dataset=is_seg_dataset,
        with_channels=with_channels,
        with_label_channels=with_label_channels,
        with_padding=with_padding,
        z_ext=z_ext,
        verify_paths=verify_paths,
    )
    return get_data_loader(ds, batch_size=batch_size, **loader_kwargs)


def default_segmentation_dataset(
    raw_paths: Union[List[Any], str, os.PathLike],
    raw_key: Optional[str],
    label_paths: Union[List[Any], str, os.PathLike],
    label_key: Optional[str],
    patch_shape: Tuple[int, ...],
    label_transform: Optional[Callable] = None,
    label_transform2: Optional[Callable] = None,
    raw_transform: Optional[Callable] = None,
    transform: Optional[Callable] = None,
    dtype: torch.dtype = torch.float32,
    label_dtype: torch.dtype = torch.float32,
    rois: Optional[Union[slice, Tuple[slice, ...]]] = None,
    n_samples: Optional[int] = None,
    sampler: Optional[Callable] = None,
    ndim: Optional[int] = None,
    is_seg_dataset: Optional[bool] = None,
    with_channels: bool = False,
    with_label_channels: bool = False,
    verify_paths: bool = True,
    with_padding: bool = True,
    z_ext: Optional[int] = None,
) -> torch.utils.data.Dataset:
    """Get data set for training a segmentation network.

    See `torch_em.data.SegmentationDataset` and `torch_em.data.ImageCollectionDataset` for details
    on the data formats that are supported.

    Args:
        raw_paths: The file path(s) to the raw data. Can either be a single path or multiple file paths.
        raw_key: The name of the internal dataset containing the raw data. Set to None for regular image files.
        label_paths: The file path(s) to the label data. Can either be a single path or multiple file paths.
        label_key: The name of the internal dataset containing the raw data. Set to None for regular image files.
        patch_shape: The patch shape for the training samples.
        label_transform: Transformation applied to the label data of a sample,
            before applying augmentations via `transform`.
        label_transform2: Transformation applied to the label data of a sample,
            after applying augmentations via `transform`.
        raw_transform: Transformation applied to the raw data of a sample,
            before applying augmentations via `transform`.
        transform: Transformation applied to both the raw data and label data of a sample.
            This can be used to implement data augmentations.
        dtype: The return data type of the raw data.
        label_dtype: The return data type of the label data.
        rois: Regions of interest in the data.  If given, the data will only be loaded from the corresponding area.
        n_samples: The length of the dataset. If None, the length will be set to `len(raw_paths)`.
        sampler: Sampler for rejecting samples according to a defined criterion.
            The sampler must be a callable that accepts the raw data (as numpy arrays) as input.
        ndim: The spatial dimensionality of the data. If None, will be derived from the raw data.
        is_seg_dataset: Whether this is a segmentation dataset or an image collection dataset.
            If None, the type of dataset will be derived from the data.
        with_channels: Whether the raw data has channels.
        with_label_channels: Whether the label data has channels.
        verify_paths: Whether to verify all paths before creating the dataset.
        with_padding: Whether to pad samples to `patch_shape` if their shape is smaller.
        z_ext: Extra bounding box for loading the data across z.
        loader_kwargs: Keyword arguments for `torch.utils.data.DataLoder`.

    Returns:
        The torch dataset.
    """
    if verify_paths:
        check_paths(raw_paths, label_paths)

    if is_seg_dataset is None:
        is_seg_dataset = is_segmentation_dataset(raw_paths, raw_key, label_paths, label_key)

    # We always use a raw transform in the convenience function.
    if raw_transform is None:
        raw_transform = get_raw_transform()

    # We always use augmentations in the convenience function.
    if transform is None:
        transform = _get_default_transform(
            raw_paths if isinstance(raw_paths, str) else raw_paths[0], raw_key, is_seg_dataset, ndim
        )

    if is_seg_dataset:
        ds = _load_segmentation_dataset(
            raw_paths,
            raw_key,
            label_paths,
            label_key,
            patch_shape=patch_shape,
            raw_transform=raw_transform,
            label_transform=label_transform,
            label_transform2=label_transform2,
            transform=transform,
            rois=rois,
            n_samples=n_samples,
            sampler=sampler,
            ndim=ndim,
            dtype=dtype,
            label_dtype=label_dtype,
            with_channels=with_channels,
            with_label_channels=with_label_channels,
            with_padding=with_padding,
            z_ext=z_ext,
        )

    else:
        ds = _load_image_collection_dataset(
            raw_paths,
            raw_key,
            label_paths,
            label_key,
            roi=rois,
            patch_shape=patch_shape,
            label_transform=label_transform,
            raw_transform=raw_transform,
            label_transform2=label_transform2,
            transform=transform,
            n_samples=n_samples,
            sampler=sampler,
            dtype=dtype,
            label_dtype=label_dtype,
            with_padding=with_padding,
        )

    return ds


def get_data_loader(dataset: torch.utils.data.Dataset, batch_size: int, **loader_kwargs) -> torch.utils.data.DataLoader:
    """@private
    """
    pin_memory = loader_kwargs.pop("pin_memory", True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, **loader_kwargs)
    # monkey patch shuffle attribute to the loader
    loader.shuffle = loader_kwargs.get("shuffle", False)
    return loader


#
# convenience functions for segmentation trainers
#


def default_segmentation_trainer(
    name: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss: Optional[torch.nn.Module] = None,
    metric: Optional[Callable] = None,
    learning_rate: float = 1e-3,
    device: Optional[Union[str, torch.device]] = None,
    log_image_interval: int = 100,
    mixed_precision: bool = True,
    early_stopping: Optional[int] = None,
    logger=TensorboardLogger,
    logger_kwargs: Optional[Dict[str, Any]] = None,
    scheduler_kwargs: Dict[str, Any] = DEFAULT_SCHEDULER_KWARGS,
    optimizer_kwargs: Dict[str, Any] = {},
    trainer_class=DefaultTrainer,
    id_: Optional[str] = None,
    save_root: Optional[str] = None,
    compile_model: Optional[Union[bool, str]] = None,
    rank: Optional[int] = None,
):
    """Get a trainer for a segmentation network.

    It creates a `torch.optim.AdamW` optimizer and learning rate scheduler that reduces the learning rate on plateau.
    By default, it uses the dice score as loss and metric.
    This can be changed by passing arguments for `loss` and/or `metric`.
    See `torch_em.trainer.DefaultTrainer` for additional details on how to configure and use the trainer.

    Here's an example for training a 2D U-Net with this function:
    ```python
    import torch_em
    from torch_em.model import UNet2d
    from torch_em.data.datasets.light_microscopy import get_dsb_loader

    # The training data will be downloaded to this location.
    data_root = "/path/to/save/the/training/data"
    patch_shape = (256, 256)
    trainer = default_segmentation_trainer(
        name="unet-training"
        model=UNet2d(in_channels=1, out_channels=1)
        train_loader=get_dsb_loader(path=data_root, patch_shape=patch_shape, split="train"),
        val_loader=get_dsb_loader(path=data_root, patch_shape=patch_shape, split="test"),
    )
    trainer.fit(iterations=int(2.5e4))  # Train for 25.000 iterations.
    ```

    Args:
        name: The name of the checkpoint that will be created by the trainer.
        model: The model to train.
        train_loader: The data loader containing the training data.
        val_loader: The data loader containing the validation data.
        loss: The loss function for training.
        metric: The metric for validation.
        learning_rate: The initial learning rate for the AdamW optimizer.
        device: The torch device to use for training. If None, will use a GPU if available.
        log_image_interval: The interval for saving images during logging, in training iterations.
        mixed_precision: Whether to train with mixed precision.
        early_stopping: The patience for early stopping in epochs. If None, early stopping will not be used.
        logger: The logger class. Will be instantiated for logging.
            By default uses `torch_em.training.tensorboard_logger.TensorboardLogger`.
        logger_kwargs: The keyword arguments for the logger class.
        scheduler_kwargs: The keyword arguments for ReduceLROnPlateau.
        optimizer_kwargs: The keyword arguments for the AdamW optimizer.
        trainer_class: The trainer class. Uses `torch_em.trainer.DefaultTrainer` by default,
            but can be set to a custom trainer class to enable custom training procedures.
        id_: Unique identifier for the trainer. If None then `name` will be used.
        save_root: The root folder for saving the checkpoint and logs.
        compile_model: Whether to compile the model before training.
        rank: Rank argument for distributed training. See `torch_em.multi_gpu_training` for details.

    Returns:
        The trainer.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, **optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)

    loss = DiceLoss() if loss is None else loss
    metric = DiceLoss() if metric is None else metric

    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(device)

    # CPU does not support mixed precision training.
    if device.type == "cpu":
        mixed_precision = False

    return trainer_class(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=metric,
        optimizer=optimizer,
        device=device,
        lr_scheduler=scheduler,
        mixed_precision=mixed_precision,
        early_stopping=early_stopping,
        log_image_interval=log_image_interval,
        logger=logger,
        logger_kwargs=logger_kwargs,
        id_=id_,
        save_root=save_root,
        compile_model=compile_model,
        rank=rank,
    )
