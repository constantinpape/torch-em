import os
import torch
from elf.io import open_file

from .data import ConcatDataset, SegmentationDataset
from .loss import DiceLoss
from .trainer import DefaultTrainer
from .trainer.tensorboard_logger import TensorboardLogger
from .transform import get_augmentations, get_raw_transform


# TODO add a heuristic to estimate this from the number of epochs
DEFAULT_SCHEDULER_KWARGS = {
    'mode': 'min',
    'factor': .5,
    'patience': 5
}


#
# convenience functions for segmentation loaders
#

# TODO implement balanced and make it the default
# def samples_to_datasets(n_samples, raw_paths, raw_key, split='balanced'):
def samples_to_datasets(n_samples, raw_paths, raw_key, split='uniform'):
    assert split in ('balanced', 'uniform')
    n_datasets = len(raw_paths)
    if split == 'uniform':
        # even distribution of samples to datasets
        samples_per_ds = n_samples // n_datasets
        divider = n_samples % n_datasets
        return [samples_per_ds + 1 if ii < divider else samples_per_ds
                for ii in range(n_datasets)]
    else:
        # distribution of samples to dataset based on the dataset lens
        raise NotImplementedError


def default_segmentation_loader(
    raw_paths,
    raw_key,
    label_paths,
    label_key,
    batch_size,
    patch_shape,
    label_transform=None,
    label_transform2=None,
    raw_transform=None,
    transform=None,
    rois=None,
    n_samples=None,
    sampler=None,
    **loader_kwargs
):
    # we always use a raw transform in the convenience function
    if raw_transform is None:
        raw_transform = get_raw_transform()

    # we always use augmentations in the convenience function
    if transform is None:
        path = raw_paths if isinstance(raw_paths, str) else raw_paths[0]
        with open_file(path, mode='r') as f:
            shape = f[raw_key].shape
            if len(shape) == 2:
                ndim = 2
            else:
                # heuristics to figure out whether to use default 3d
                # or default anisotropic augmentations
                ndim = 'anisotropic' if shape[0] < shape[1] // 2 else 3
        transform = get_augmentations(ndim)

    if isinstance(raw_paths, str):
        assert isinstance(label_paths, str)
        assert os.path.exists(raw_paths)
        assert os.path.exists(label_paths)
        if rois is not None:
            assert len(rois) == 3 and all(isinstance(roi, slice) for roi in rois)
        ds = SegmentationDataset(
            raw_paths, raw_key,
            label_paths, label_key,
            patch_shape=patch_shape,
            raw_transform=raw_transform,
            label_transform=label_transform,
            label_transform2=label_transform2,
            transform=transform,
            roi=rois,
            n_samples=n_samples,
            sampler=sampler
        )
    else:
        assert len(raw_paths) == len(label_paths)
        assert all(os.path.exists(rp) for rp in raw_paths)
        assert all(os.path.exists(lp) for lp in raw_paths)
        if rois is not None:
            assert len(rois) == len(label_paths)
            assert all(isinstance(roi, tuple) for roi in rois)

        samples_per_ds = [None] * len(raw_paths) if n_samples is None else samples_to_datasets(n_samples,
                                                                                               raw_paths,
                                                                                               raw_key)
        ds = []
        for i, (raw_path, label_path) in enumerate(zip(raw_paths, label_paths)):
            roi = None if rois is None else rois[i]
            dset = SegmentationDataset(
                raw_path, raw_key,
                label_path, label_key,
                patch_shape=patch_shape,
                raw_transform=raw_transform,
                label_transform=label_transform,
                label_transform2=label_transform2,
                transform=transform,
                roi=roi,
                n_samples=samples_per_ds[i],
                sampler=sampler
            )
            ds.append(dset)
        ds = ConcatDataset(*ds)

    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, **loader_kwargs)
    return loader


#
# convenience functions for segmentation trainers
#


def default_segmentation_trainer(
    name,
    model,
    train_loader,
    val_loader,
    loss=None,
    metric=None,
    learning_rate=1e-3,
    device=None,
    log_image_interval=100,
    mixed_precision=True,
    early_stopping=None,
    logger=TensorboardLogger,
    scheduler_kwargs=DEFAULT_SCHEDULER_KWARGS,
    optimizer_kwargs={}
):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 **optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        **scheduler_kwargs
    )

    loss = DiceLoss() if loss is None else loss
    metric = DiceLoss() if metric is None else metric

    if device is None:
        device = torch.device('cuda')

    trainer = DefaultTrainer(
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
        logger=logger
    )
    return trainer
