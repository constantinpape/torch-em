import time
from copy import deepcopy

import torch
import torch.cuda.amp as amp
from .default_trainer import DefaultTrainer


# TODO over-ride from_checkpoint, load_checkpoint to load the model
class SPOCOTrainer(DefaultTrainer):
    def __init__(self, momentum=0.999, **kwargs):
        super().__init__(**kwargs)
        self.momentum = momentum
        # copy the model and don"t require gradients for it
        self.model2 = deepcopy(self.model)
        for param in self.model2.parameters():
            param.requires_grad = False

    def _momentum_update(self):
        for param1, param2 in zip(self.model.parameters(), self.model2.parameters()):
            param2.data = param1.data * self.momentum + param2.data * (1. - self.momentum)

    def save_checkpoint(self, name, best_metric):
        model2_state = {"model2_state": self.model2.state()}
        super().save_checkpoint(name, best_metric, model2_state)

    def _train_epoch(self, progress):
        self.model.train()

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

        t_per_iter = (time.time() - t_per_iter) / n_iter
        return t_per_iter

    def _train_epoch_mixed(self, progress):
        self.model.train()

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

        t_per_iter = (time.time() - t_per_iter) / n_iter
        return t_per_iter

    def _validate(self):
        self.model.eval()

        metric = 0.
        loss = 0.

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

        metric_val = 0.
        loss_val = 0.

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
