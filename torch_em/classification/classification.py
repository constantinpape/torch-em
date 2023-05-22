from functools import partial

import sklearn.metrics as metrics
import torch
import torch_em

from .classification_dataset import ClassificationDataset
from .classification_logger import ClassificationLogger
from .classification_trainer import ClassificationTrainer


class ClassificationMetric:
    def __init__(self, metric_name="accuracy_score", **metric_kwargs):
        if not hasattr(metrics, metric_name):
            raise ValueError(f"Invalid metric_name {metric_name}")
        self.metric = getattr(metrics, metric_name)
        self.metric_kwargs = metric_kwargs

    def __call__(self, y_true, y_pred):
        metric_error = 1.0 - self.metric(y_true, y_pred, **self.metric_kwargs)
        return metric_error


def default_classification_loader(
    data, target, batch_size, normalization=None, augmentation=None, image_shape=None, **loader_kwargs,
):
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


# TODO
def zarr_classification_loader():
    return default_classification_loader()


def default_classification_trainer(
    name,
    model,
    train_loader,
    val_loader,
    loss=None,
    metric=None,
    logger=ClassificationLogger,
    trainer_class=ClassificationTrainer,
    **kwargs,
):
    """
    """
    # set the default loss and metric (if no values where passed)
    loss = torch.nn.CrossEntropyLoss() if loss is None else loss
    metric = ClassificationMetric() if metric is None else metric

    # metric: note that we use lower metric = better !
    # so we record the accuracy error instead of the error rate
    trainer = torch_em.default_segmentation_trainer(
        name, model, train_loader, val_loader,
        loss=loss, metric=metric,
        logger=logger, trainer_class=trainer_class,
        **kwargs,
    )
    return trainer
