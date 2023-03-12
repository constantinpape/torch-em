import time
from copy import deepcopy

import torch
import torch_em

from .logger import SelfTrainingTensorboardLogger


class Dummy(torch.nn.Module):
    pass


class MeanTeacherTrainer(torch_em.trainer.DefaultTrainer):
    """This trainer implements self-traning for semi-supervised learning and domain following the 'MeanTeacher' approach
    of Tarvainen & Vapola (https://arxiv.org/abs/1703.01780). This approach uses a teacher model derived from the
    student model via EMA of weights to predict pseudo-labels on unlabeled data.
    We support two training strategies: joint training on labeled and unlabeled data
    (with a supervised and unsupervised loss function). And training only on the unsupervised data.

    This class expects the following data loaders:
    - unsupervised_train_loader: Returns two augmentations of the same input.
    - supervised_train_loader (optional): Returns input and labels.
    - unsupervised_val_loader (optional): Same as unsupervised_train_loader
    - supervised_val_loader (optional): Same as supervised_train_loader
    At least one of unsupervised_val_loader and supervised_val_loader must be given.

    And the following elements to customize the pseudo labeling:
    - pseudo_labeler: to compute the psuedo-labels
        - Parameters: teacher, teacher_input
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

    If the parameter reinit_teacher is set to true, the teacher weights are re-initialized.
    If it is None, the most appropriate initialization scheme for the training approach is chosen:
    - semi-supervised training -> reinit, because we usually train a model from scratch
    - unsupervised training -> do not reinit, because we usually fine-tune a model

    Note: adjust the batch size ratio between the 'unsupervised_train_loader' and 'supervised_train_loader'
    for setting the ratio between supervised and unsupervised training samples

    Parameters:
        model [nn.Module] -
        unsupervised_train_loader [torch.DataLoader] -
        unsupervised_loss [callable] -
        supervised_train_loader [torch.DataLoader] - (default: None)
        supervised_loss [callable] - (default: None)
        unsupervised_loss_and_metric [callable] - (default: None)
        supervised_loss_and_metric [callable] - (default: None)
        logger [TorchEmLogger] - (default: SelfTrainingTensorboardLogger)
        momentum [float] - (default: 0.999)
        reinit_teacher [bool] - (default: None)
        **kwargs - keyword arguments for torch_em.DataLoader
    """

    def __init__(
        self,
        model,
        unsupervised_train_loader,
        unsupervised_loss,
        pseudo_labeler,
        supervised_train_loader=None,
        unsupervised_val_loader=None,
        supervised_val_loader=None,
        supervised_loss=None,
        unsupervised_loss_and_metric=None,
        supervised_loss_and_metric=None,
        logger=SelfTrainingTensorboardLogger,
        momentum=0.999,
        reinit_teacher=None,
        **kwargs
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

        super().__init__(
            model=model, train_loader=train_loader, val_loader=val_loader,
            loss=Dummy(), metric=Dummy(), logger=logger, **kwargs
        )

        self.unsupervised_loss = unsupervised_loss
        self.supervised_loss = supervised_loss

        self.pseudo_labeler = pseudo_labeler

        self.momentum = momentum

        # determine how we initialize the teacher weights (copy or reinitialization)
        if reinit_teacher is None:
            # semisupervised training: reinitialize
            # unsupervised training: copy
            self.reinit_teacher = supervised_train_loader is not None
        else:
            self.reinit_teacher = reinit_teacher

        with torch.no_grad():
            self.teacher = deepcopy(self.model)
            if self.reinit_teacher:
                for layer in self.teacher.children():
                    if hasattr(layer, "reset_parameters"):
                        layer.reset_parameters()
            for param in self.teacher.parameters():
                param.requires_grad = False

        self._kwargs = kwargs

    def _momentum_update(self):
        # if we reinit the teacher we perform much faster updates (low momentum) in the first iterations
        # to avoid a large gap between teacher and student weights, leading to inconsistent predictions
        # if we don't reinit this is not necessary
        if self.reinit_teacher:
            current_momentum = min(1 - 1 / (self._iteration + 1), self.momentum)
        else:
            current_momentum = self.momentum

        for param, param_teacher in zip(self.model.parameters(), self.teacher.parameters()):
            param_teacher.data = param_teacher.data * current_momentum + param.data * (1. - current_momentum)

    # TODO I think we need to serialize more things here for all the loaders etc.
    #
    # functionality for saving checkpoints and initialization
    #

    def save_checkpoint(self, name, best_metric):
        teacher_state = {"teacher_state": self.teacher.state_dict()}
        super().save_checkpoint(name, best_metric, **teacher_state)

    def load_checkpoint(self, checkpoint="best"):
        save_dict = super().load_checkpoint(checkpoint)
        self.teacher.load_state_dict(save_dict["teacher_state"])
        self.teacher.to(self.device)
        return save_dict

    def _initialize(self, iterations, load_from_checkpoint):
        best_metric = super()._initialize(iterations, load_from_checkpoint)
        self.teacher.to(self.device)
        return best_metric

    #
    # training and validation functionality
    #

    def _train_epoch_unsupervised(self, progress, forward_context, backprop):
        self.model.train()

        n_iter = 0
        t_per_iter = time.time()

        # Sample from both the supervised and unsupervised loader.
        for xu1, xu2 in self.unsupervised_train_loader:
            xu1, xu2 = xu1.to(self.device), xu2.to(self.device)

            teacher_input, model_input = xu1, xu2
            # Perform unsupervised training
            with forward_context():
                # Compute the pseudo labels.
                pseudo_labels, label_filter = self.pseudo_labeler(self.teacher, teacher_input)
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

            with torch.no_grad():
                self._momentum_update()

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
            xs, ys = xs.to(self.device), ys.to(self.device)
            xu1, xu2 = xu1.to(self.device), xu2.to(self.device)

            # Perform supervised training.
            self.optimizer.zero_grad()
            with forward_context():
                # We pass the model, the input and the labels to the supervised loss function,
                # so that how the loss is calculated stays flexible, e.g. to enable ELBO for PUNet.
                supervised_loss = self.supervised_loss(self.model, xs, ys)

            teacher_input, model_input = xu1, xu2
            # Perform unsupervised training
            with forward_context():
                pseudo_labels, label_filter = self.pseudo_labeler(self.teacher, teacher_input)
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

            with torch.no_grad():
                self._momentum_update()

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
            x, y = x.to(self.device), y.to(self.device)
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
            x1, x2 = x1.to(self.device), x2.to(self.device)
            teacher_input, model_input = x1, x2
            with forward_context():
                pseudo_labels, label_filter = self.pseudo_labeler(self.teacher, teacher_input)
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
