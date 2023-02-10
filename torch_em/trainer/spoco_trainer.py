import time
from copy import deepcopy

import torch
import torch.cuda.amp as amp
from .default_trainer import DefaultTrainer


class SPOCOTrainer(DefaultTrainer):
    def __init__(
        self,
        model,
        momentum=0.999,
        semisupervised_loss=None,
        semisupervised_loader=None,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
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

    def save_checkpoint(self, name, best_metric):
        model2_state = {"model2_state": self.model2.state_dict()}
        super().save_checkpoint(name, best_metric, **model2_state)

    def load_checkpoint(self, checkpoint="best"):
        save_dict = super().load_checkpoint(checkpoint)
        self.model2.load_state_dict(save_dict["model2_state"])
        self.model2.to(self.device)
        return save_dict

    def _initialize(self, iterations, load_from_checkpoint):
        best_metric = super()._initialize(iterations, load_from_checkpoint)
        self.model2.to(self.device)
        return best_metric

    def _train_epoch_semisupervised(self, progress):
        self.model.train()
        self.model2.train()
        progress.set_description(
            f"Run semi-supervised training for {len(self.semisupervised_loader)} iterations", refresh=True
        )

        for x in self.semisupervised_loader:
            x = x.to(self.device)
            self.optimizer.zero_grad()

            prediction = self.model(x)
            with torch.no_grad():
                self._momentum_update()
                prediction2 = self.model2(x)
            loss = self.semisupervised_loss(prediction, prediction2)
            loss.backward()
            self.optimizer.step()

    def _train_epoch(self, progress):
        self.model.train()
        self.model2.train()

        n_iter = 0
        t_per_iter = time.time()
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            prediction = self.model(x)
            with torch.no_grad():
                self._momentum_update()
                prediction2 = self.model2(x)
            if self._iteration % self.log_image_interval == 0:
                prediction.retain_grad()
            loss = self.loss((prediction, prediction2), y)

            loss.backward()
            self.optimizer.step()

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
            self._train_epoch_semisupervised(progress)
        t_per_iter = (time.time() - t_per_iter) / n_iter
        return t_per_iter

    def _train_epoch_semisupervised_mixed(self, progress):
        self.model.train()
        self.model2.train()
        progress.set_description(
            f"Run semi-supervised training for {len(self.semisupervised_loader)} iterations", refresh=True
        )

        for x in self.semisupervised_loader:
            x = x.to(self.device)
            self.optimizer.zero_grad()

            with amp.autocast():
                prediction = self.model(x)
                with torch.no_grad():
                    self._momentum_update()
                    prediction2 = self.model2(x)
                loss = self.semisupervised_loss(prediction, prediction2)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

    def _train_epoch_mixed(self, progress):
        self.model.train()
        self.model2.train()

        n_iter = 0
        t_per_iter = time.time()
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            with amp.autocast():
                prediction = self.model(x)
                with torch.no_grad():
                    self._momentum_update()
                    prediction2 = self.model2(x)
                loss = self.loss((prediction, prediction2), y)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            lr = [pm["lr"] for pm in self.optimizer.param_groups][0]
            if self.logger is not None:
                self.logger.log_train(self._iteration, loss, lr,
                                      x, y, prediction)

            self._iteration += 1
            n_iter += 1
            if self._iteration >= self.max_iteration:
                break
            progress.update(1)

        if self.semisupervised_loader is not None:
            self._train_epoch_semisupervised_mixed(progress)
        t_per_iter = (time.time() - t_per_iter) / n_iter
        return t_per_iter

    def _validate(self):
        self.model.eval()
        self.model2.eval()

        metric = 0.0
        loss = 0.0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                prediction = self.model(x)
                prediction2 = self.model2(x)
                loss += self.loss((prediction, prediction2), y).item()
                metric += self.metric(prediction, y).item()

        metric /= len(self.val_loader)
        loss /= len(self.val_loader)
        if self.logger is not None:
            self.logger.log_validation(self._iteration, metric, loss,
                                       x, y, prediction)
        return metric

    def _validate_mixed(self):
        self.model.eval()
        self.model2.eval()

        metric_val = 0.0
        loss_val = 0.0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                with amp.autocast():
                    prediction = self.model(x)
                    prediction2 = self.model2(x)
                    loss = self.loss((prediction, prediction2), y)
                    metric = self.metric(prediction, y)
                loss_val += loss
                metric_val += metric

        metric_val /= len(self.val_loader)
        loss_val /= len(self.val_loader)
        if self.logger is not None:
            self.logger.log_validation(self._iteration, metric, loss,
                                       x, y, prediction)
        return metric_val
