import time
from typing import Callable, List, Optional

import torch
import torch_em
from torch_em.util import get_constructor_arguments

from .logger import SelfTrainingTensorboardLogger
from .mean_teacher import Dummy


class FixMatchTrainer(torch_em.trainer.DefaultTrainer):
    """Trainer for semi-supervised learning and domain adaptation following the FixMatch approach.

    FixMatch was introduced by Sohn et al. in https://arxiv.org/abs/2001.07685).
    It uses a teacher model derived from the student model via weight sharing to predict pseudo-labels
    on unlabeled data. We support two training strategies:
    - Joint training on labeled and unlabeled data (with a supervised and unsupervised loss function).
    - Taining only on the unsupervised data.

    This class expects the following data loaders:
    - unsupervised_train_loader: Returns two augmentations (weak and strong) of the same input.
    - supervised_train_loader (optional): Returns input and labels.
    - unsupervised_val_loader (optional): Same as unsupervised_train_loader
    - supervised_val_loader (optional): Same as supervised_train_loader
    At least one of unsupervised_val_loader and supervised_val_loader must be given.

    The following arguments can be used to customize the pseudo labeling:
    - pseudo_labeler: to compute the psuedo-labels
        - Parameters: model, teacher_input
        - Returns: pseudo_labels, label_filter (<- label filter can for example be mask, weight or None)
    - unsupervised_loss: the loss between model predictions and pseudo labels
        - Parameters: model, model_input, pseudo_labels, label_filter
        - Returns: loss
    - supervised_loss (optional): the supervised loss function
        - Parameters: model, input, labels
        - Returns: loss
    - unsupervised_loss_and_metric (optional): the unsupervised loss function and metric
        - Parameters: model, model_input, pseudo_labels, label_filter
        - Returns: loss, metric
    - supervised_loss_and_metric (optional): the supervised loss function and metric
        - Parameters: model, input, labels
        - Returns: loss, metric
    At least one of unsupervised_loss_and_metric and supervised_loss_and_metric must be given.

    Note: adjust the batch size ratio between the 'unsupervised_train_loader' and 'supervised_train_loader'
    for setting the ratio between supervised and unsupervised training samples.

    Args:
        model: The model to be trained.
        unsupervised_train_loader: The loader for unsupervised training.
        unsupervised_loss: The loss for unsupervised training.
        pseudo_labeler: The pseudo labeler that predicts labels in unsupervised training.
        supervised_train_loader: The loader for supervised training.
        supervised_loss: The loss for supervised training.
        unsupervised_loss_and_metric: The loss and metric for unsupervised training.
        supervised_loss_and_metric: The loss and metrhic for supervised training.
        logger: The logger.
        source_distribution: The ratio of labels in the source label distribution.
            If given, the predicted distribution of the trained model will be regularized to
            match this source label distribution.
        kwargs: Additional keyword arguments for `torch_em.trainer.DefaultTrainer`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        unsupervised_train_loader: torch.utils.data.DataLoader,
        unsupervised_loss: torch.utils.data.DataLoader,
        pseudo_labeler: Callable,
        supervised_train_loader: Optional[torch.utils.data.DataLoader] = None,
        unsupervised_val_loader: Optional[torch.utils.data.DataLoader] = None,
        supervised_val_loader: Optional[torch.utils.data.DataLoader] = None,
        supervised_loss: Optional[Callable] = None,
        unsupervised_loss_and_metric: Optional[Callable] = None,
        supervised_loss_and_metric: Optional[Callable] = None,
        logger=SelfTrainingTensorboardLogger,
        source_distribution: List[float] = None,
        **kwargs,
    ):
        # Do we have supervised data or not?
        if supervised_train_loader is None:
            # No. -> We use the unsupervised training logic.
            train_loader = unsupervised_train_loader
            self._train_epoch_impl = self._train_epoch_unsupervised
        else:
            # Yes. -> We use the semi-supervised training logic.
            assert supervised_loss is not None
            train_loader = supervised_train_loader if len(supervised_train_loader) < len(unsupervised_train_loader)\
                else unsupervised_train_loader
            self._train_epoch_impl = self._train_epoch_semisupervised

        self.unsupervised_train_loader = unsupervised_train_loader
        self.supervised_train_loader = supervised_train_loader

        # Check that we have at least one of supvervised / unsupervised val loader.
        assert sum((
            supervised_val_loader is not None,
            unsupervised_val_loader is not None,
        )) > 0
        self.supervised_val_loader = supervised_val_loader
        self.unsupervised_val_loader = unsupervised_val_loader

        if self.unsupervised_val_loader is None:
            val_loader = self.supervised_val_loader
        else:
            val_loader = self.unsupervised_train_loader

        # Check that we have at least one of supvervised / unsupervised loss and metric.
        assert sum((
            supervised_loss_and_metric is not None,
            unsupervised_loss_and_metric is not None,
        )) > 0
        self.supervised_loss_and_metric = supervised_loss_and_metric
        self.unsupervised_loss_and_metric = unsupervised_loss_and_metric

        # train_loader, val_loader, loss and metric may be unnecessarily deserialized
        kwargs.pop("train_loader", None)
        kwargs.pop("val_loader", None)
        kwargs.pop("metric", None)
        kwargs.pop("loss", None)
        super().__init__(
            model=model, train_loader=train_loader, val_loader=val_loader,
            loss=Dummy(), metric=Dummy(), logger=logger, **kwargs
        )

        self.unsupervised_loss = unsupervised_loss
        self.supervised_loss = supervised_loss

        self.pseudo_labeler = pseudo_labeler

        if source_distribution is None:
            self.source_distribution = None
        else:
            self.source_distribution = torch.FloatTensor(source_distribution).to(self.device)

        self._kwargs = kwargs

    #
    # functionality for saving checkpoints and initialization
    #

    def save_checkpoint(self, name, current_metric, best_metric, **extra_save_dict):
        """@private
        """
        train_loader_kwargs = get_constructor_arguments(self.train_loader)
        val_loader_kwargs = get_constructor_arguments(self.val_loader)
        extra_state = {
            "init": {
                "train_loader_kwargs": train_loader_kwargs,
                "train_dataset": self.train_loader.dataset,
                "val_loader_kwargs": val_loader_kwargs,
                "val_dataset": self.val_loader.dataset,
                "loss_class": "torch_em.self_training.mean_teacher.Dummy",
                "loss_kwargs": {},
                "metric_class": "torch_em.self_training.mean_teacher.Dummy",
                "metric_kwargs": {},
            },
        }
        extra_state.update(**extra_save_dict)
        super().save_checkpoint(name, current_metric, best_metric, **extra_state)

    # Distribution alignment:
    # Encourages the distribution of the model's generated pseudo labels to match the marginal
    # distribution of pseudo labels from the source transfer (key idea: to maximize the mutual information).
    def get_distribution_alignment(self, pseudo_labels, label_threshold=0.5):
        """@private
        """
        if self.source_distribution is not None:
            pseudo_labels_binary = torch.where(pseudo_labels >= label_threshold, 1, 0)
            _, target_distribution = torch.unique(pseudo_labels_binary, return_counts=True)
            target_distribution = target_distribution / target_distribution.sum()
            distribution_ratio = self.source_distribution / target_distribution
            pseudo_labels = torch.where(
                pseudo_labels < label_threshold,
                pseudo_labels * distribution_ratio[0],
                pseudo_labels * distribution_ratio[1]
            ).clip(0, 1)

        return pseudo_labels

    #
    # training and validation functionality
    #

    def _train_epoch_unsupervised(self, progress, forward_context, backprop):
        self.model.train()

        n_iter = 0
        t_per_iter = time.time()

        # Sample from both the supervised and unsupervised loader.
        for xu1, xu2 in self.unsupervised_train_loader:
            xu1, xu2 = xu1.to(self.device, non_blocking=True), xu2.to(self.device, non_blocking=True)

            teacher_input, model_input = xu1, xu2

            with forward_context(), torch.no_grad():
                # Compute the pseudo labels.
                pseudo_labels, label_filter = self.pseudo_labeler(self.model, teacher_input)

            pseudo_labels = pseudo_labels.detach()
            if label_filter is not None:
                label_filter = label_filter.detach()

            # Perform distribution alignment for pseudo labels
            pseudo_labels = self.get_distribution_alignment(pseudo_labels)

            self.optimizer.zero_grad()
            # Perform unsupervised training
            with forward_context():
                loss = self.unsupervised_loss(self.model, model_input, pseudo_labels, label_filter)

            backprop(loss)

            if self.logger is not None:
                with torch.no_grad(), forward_context():
                    pred = self.model(model_input) if self._iteration % self.log_image_interval == 0 else None
                self.logger.log_train_unsupervised(
                    self._iteration, loss, xu1, xu2, pred, pseudo_labels, label_filter
                )
                lr = [pm["lr"] for pm in self.optimizer.param_groups][0]
                self.logger.log_lr(self._iteration, lr)

            self._iteration += 1
            n_iter += 1
            if self._iteration >= self.max_iteration:
                break
            progress.update(1)

        t_per_iter = (time.time() - t_per_iter) / n_iter
        return t_per_iter

    def _train_epoch_semisupervised(self, progress, forward_context, backprop):
        self.model.train()

        n_iter = 0
        t_per_iter = time.time()

        # Sample from both the supervised and unsupervised loader.
        for (xs, ys), (xu1, xu2) in zip(self.supervised_train_loader, self.unsupervised_train_loader):
            xs, ys = xs.to(self.device, non_blocking=True), ys.to(self.device, non_blocking=True)
            xu1, xu2 = xu1.to(self.device, non_blocking=True), xu2.to(self.device, non_blocking=True)

            # Perform supervised training.
            self.optimizer.zero_grad()
            with forward_context():
                # We pass the model, the input and the labels to the supervised loss function,
                # so that how the loss is calculated stays flexible, e.g. to enable ELBO for PUNet.
                supervised_loss = self.supervised_loss(self.model, xs, ys)

            teacher_input, model_input = xu1, xu2

            with forward_context(), torch.no_grad():
                # Compute the pseudo labels.
                pseudo_labels, label_filter = self.pseudo_labeler(self.model, teacher_input)

            pseudo_labels = pseudo_labels.detach()
            if label_filter is not None:
                label_filter = label_filter.detach()

            # Perform distribution alignment for pseudo labels
            pseudo_labels = self.get_distribution_alignment(pseudo_labels)

            # Perform unsupervised training
            with forward_context():
                unsupervised_loss = self.unsupervised_loss(self.model, model_input, pseudo_labels, label_filter)

            loss = (supervised_loss + unsupervised_loss) / 2
            backprop(loss)

            if self.logger is not None:
                with torch.no_grad(), forward_context():
                    unsup_pred = self.model(model_input) if self._iteration % self.log_image_interval == 0 else None
                    supervised_pred = self.model(xs) if self._iteration % self.log_image_interval == 0 else None

                self.logger.log_train_supervised(self._iteration, supervised_loss, xs, ys, supervised_pred)
                self.logger.log_train_unsupervised(
                    self._iteration, unsupervised_loss, xu1, xu2, unsup_pred, pseudo_labels, label_filter
                )

                self.logger.log_combined_loss(self._iteration, loss)
                lr = [pm["lr"] for pm in self.optimizer.param_groups][0]
                self.logger.log_lr(self._iteration, lr)

            self._iteration += 1
            n_iter += 1
            if self._iteration >= self.max_iteration:
                break
            progress.update(1)

        t_per_iter = (time.time() - t_per_iter) / n_iter
        return t_per_iter

    def _validate_supervised(self, forward_context):
        metric_val = 0.0
        loss_val = 0.0

        for x, y in self.supervised_val_loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with forward_context():
                loss, metric = self.supervised_loss_and_metric(self.model, x, y)
            loss_val += loss.item()
            metric_val += metric.item()

        metric_val /= len(self.supervised_val_loader)
        loss_val /= len(self.supervised_val_loader)

        if self.logger is not None:
            with forward_context():
                pred = self.model(x)
            self.logger.log_validation_supervised(self._iteration, metric_val, loss_val, x, y, pred)

        return metric_val

    def _validate_unsupervised(self, forward_context):
        metric_val = 0.0
        loss_val = 0.0

        for x1, x2 in self.unsupervised_val_loader:
            x1, x2 = x1.to(self.device, non_blocking=True), x2.to(self.device, non_blocking=True)
            teacher_input, model_input = x1, x2
            with forward_context():
                pseudo_labels, label_filter = self.pseudo_labeler(self.model, teacher_input)
                loss, metric = self.unsupervised_loss_and_metric(self.model, model_input, pseudo_labels, label_filter)
            loss_val += loss.item()
            metric_val += metric.item()

        metric_val /= len(self.unsupervised_val_loader)
        loss_val /= len(self.unsupervised_val_loader)

        if self.logger is not None:
            with forward_context():
                pred = self.model(model_input)
            self.logger.log_validation_unsupervised(
                self._iteration, metric_val, loss_val, x1, x2, pred, pseudo_labels, label_filter
            )

        return metric_val

    def _validate_impl(self, forward_context):
        self.model.eval()

        with torch.no_grad():

            if self.supervised_val_loader is None:
                supervised_metric = None
            else:
                supervised_metric = self._validate_supervised(forward_context)

            if self.unsupervised_val_loader is None:
                unsupervised_metric = None
            else:
                unsupervised_metric = self._validate_unsupervised(forward_context)

        if unsupervised_metric is None:
            metric = supervised_metric
        elif supervised_metric is None:
            metric = unsupervised_metric
        else:
            metric = (supervised_metric + unsupervised_metric) / 2

        return metric
