from functools import partial
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import sklearn.metrics as metrics
import torch
import torch_em
from numpy.typing import ArrayLike

from .classification_dataset import ClassificationDataset
from .classification_logger import ClassificationLogger
from .classification_trainer import ClassificationTrainer


class ClassificationMetric:
    """Metric for classification training.

    Args:
        metric_name: The name of the metrics. The name will be looked up in `sklearn.metrics`,
            so it must be a valid identifier in that python package.
        metric_kwargs: Keyword arguments for the metric.
    """
    def __init__(self, metric_name: str = "accuracy_score", **metric_kwargs):
        if not hasattr(metrics, metric_name):
            raise ValueError(f"Invalid metric_name {metric_name}.")
        self.metric = getattr(metrics, metric_name)
        self.metric_kwargs = metric_kwargs

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Evaluate model prediction against classification labels.

        Args:
            y_true: The classification labels.
            y_pred: The model predictions.

        Returns:
            The metric value.
        """
        metric_error = 1.0 - self.metric(y_true, y_pred, **self.metric_kwargs)
        return metric_error


def default_classification_loader(
    data: Sequence[ArrayLike],
    target: Sequence[ArrayLike],
    batch_size: int,
    normalization: Optional[callable] = None,
    augmentation: Optional[callable] = None,
    image_shape: Optional[Tuple[int, ...]] = None,
    **loader_kwargs,
) -> torch.utils.data.DataLoader:
    """Get a data loader for classification training.

    Args:
        data: The input data for classification. Expects a sequence of array-like data.
            The data can be two or three dimensional.
        target: The target data for classification. Expects a sequence of the same length as `data`.
            Each value in the sequence must be a scalar.
        batch_size: The batch size for the data loader.
        normalization: The normalization function. If None, data standardization will be used.
        augmentation: The augmentation function. If None, the default augmentations will be used.
        image_shape: The target shape of the data. If given, each sample will be resampled to this size.
        loader_kwargs: Additional keyword arguments for `torch.utils.data.DataLoader`.

    Returns:
        The data loader.
    """
    ndim = data[0].ndim - 1
    if ndim not in (2, 3):
        raise ValueError(f"Expect input data of dimensionality 2 or 3, got {ndim}")

    if normalization is None:
        axis = (1, 2) if ndim == 2 else (1, 2, 3)
        normalization = partial(torch_em.transform.raw.standardize, axis=axis)

    if augmentation is None:
        augmentation = torch_em.transform.get_augmentations(ndim=ndim)

    dataset = ClassificationDataset(data, target, normalization, augmentation, image_shape)
    loader = torch_em.segmentation.get_data_loader(dataset, batch_size, **loader_kwargs)
    return loader


def default_classification_trainer(
    name: str,
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    loss: Optional[Union[torch.nn.Module, callable]] = None,
    metric: Optional[Union[torch.nn.Module, callable]] = None,
    logger=ClassificationLogger,
    trainer_class=ClassificationTrainer,
    **kwargs,
):
    """Get a trainer for a classification task.

    This will create an instance of `torch_em.classification.ClassificationTrainer`.
    Check out its documentation string for details on how to configure and use the trainer.

    Args:
        name: The name for the checkpoint created by the trainer.
        model: The classification model to train.
        train_loader: The data loader for training.
        val_loader: The data loader for validation.
        loss: The loss function. If None, will use cross entropy.
        metric: The metric function. If None, will use the accuracy error.
        logger: The logger for keeping track of the training progress.
        trainer_class: The trainer class.
        kwargs: Keyword arguments for the trainer class.

    Returns:
        The classification trainer.
    """
    # Set the default loss and metric (if no values where passed).
    loss = torch.nn.CrossEntropyLoss() if loss is None else loss
    metric = ClassificationMetric() if metric is None else metric

    # Metric: Note that we use lower metric = better.
    # So we record the accuracy error instead of the accuracy..
    trainer = torch_em.default_segmentation_trainer(
        name, model, train_loader, val_loader,
        loss=loss, metric=metric,
        logger=logger, trainer_class=trainer_class,
        **kwargs,
    )
    return trainer
