import time
from copy import deepcopy
from typing import Optional

import torch
from .default_trainer import DefaultTrainer
from .tensorboard_logger import TensorboardLogger


class SPOCOTrainer(DefaultTrainer):
    """Trainer for a SPOCO model.

    For details check out "Sparse Object-level Supervision for Instance Segmentation with Pixel Embeddings":
    https://arxiv.org/abs/2103.14572

    Args:
        model: The model to train.
        momentum: The momementum value for exponential moving weight averaging.
        semisupervised_loss: Optional loss for semi-supervised learning.
        semisupervised_loader: Optional data loader for semi-supervised learning.
        logger: The logger.
        kwargs: Additional keyord arguments for `torch_em.trainer.DefaultTrainer`.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        momentum: float = 0.999,
        semisupervised_loss: Optional[torch.nn.Module] = None,
        semisupervised_loader: Optional[torch.utils.data.DataLoader] = None,
        logger=TensorboardLogger,
        **kwargs,
    ):
        super().__init__(model=model, logger=logger, **kwargs)
        self.momentum = momentum
        # copy the model and don"t require gradients for it
        self.model2 = deepcopy(self.model)
        for param in self.model2.parameters():
            param.requires_grad = False
        # do we have a semi-supervised loss and loader?
        assert (semisupervised_loss is None) == (semisupervised_loader is None)
        self.semisupervised_loader = semisupervised_loader
        self.semisupervised_loss = semisupervised_loss
        self._kwargs = kwargs

    def _momentum_update(self):
        for param_model, param_teacher in zip(self.model.parameters(), self.model2.parameters()):
            param_teacher.data = param_teacher.data * self.momentum + param_model.data * (1. - self.momentum)

    def save_checkpoint(self, name, current_metric, best_metric, **extra_save_dict):
        """@private
        """
        super().save_checkpoint(
            name, current_metric, best_metric, model2_state=self.model2.state_dict(), **extra_save_dict
        )

    def load_checkpoint(self, checkpoint="best"):
        """@private
        """
        save_dict = super().load_checkpoint(checkpoint)
        self.model2.load_state_dict(save_dict["model2_state"])
        self.model2.to(self.device)
        return save_dict

    def _initialize(self, iterations, load_from_checkpoint, epochs=None):
        best_metric = super()._initialize(iterations, load_from_checkpoint, epochs)
        self.model2.to(self.device)
        return best_metric

    def _train_epoch_semisupervised(self, progress, forward_context, backprop):
        self.model.train()
        self.model2.train()
        progress.set_description(
            f"Run semi-supervised training for {len(self.semisupervised_loader)} iterations", refresh=True
        )

        for x in self.semisupervised_loader:
            x = x.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()

            with forward_context():
                prediction = self.model(x)
                with torch.no_grad():
                    prediction2 = self.model2(x)
                loss = self.semisupervised_loss(prediction, prediction2)
            backprop(loss)

            with torch.no_grad():
                self._momentum_update()

    def _train_epoch_impl(self, progress, forward_context, backprop):
        self.model.train()
        self.model2.train()

        n_iter = 0
        t_per_iter = time.time()
        for x, y in self.train_loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with forward_context():
                prediction = self.model(x)
                with torch.no_grad():
                    prediction2 = self.model2(x)
                loss = self.loss((prediction, prediction2), y)

            if self._iteration % self.log_image_interval == 0:
                prediction.retain_grad()

            backprop(loss)

            with torch.no_grad():
                self._momentum_update()

            lr = [pm["lr"] for pm in self.optimizer.param_groups][0]
            if self.logger is not None:
                self.logger.log_train(self._iteration, loss, lr,
                                      x, y, prediction,
                                      log_gradients=True)

            self._iteration += 1
            n_iter += 1
            if self._iteration >= self.max_iteration:
                break
            progress.update(1)

        if self.semisupervised_loader is not None:
            self._train_epoch_semisupervised(progress, forward_context, backprop)
        t_per_iter = (time.time() - t_per_iter) / n_iter
        return t_per_iter

    def _validate_impl(self, forward_context):
        self.model.eval()
        self.model2.eval()

        metric = 0.0
        loss = 0.0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                with forward_context():
                    prediction = self.model(x)
                    prediction2 = self.model2(x)
                loss += self.loss((prediction, prediction2), y).item()
                metric += self.metric(prediction, y).item()

        metric /= len(self.val_loader)
        loss /= len(self.val_loader)
        if self.logger is not None:
            self.logger.log_validation(self._iteration, metric, loss, x, y, prediction)
        return metric
