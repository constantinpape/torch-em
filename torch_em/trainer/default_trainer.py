import os
import time
import warnings

import numpy as np
import torch
import torch.cuda.amp as amp
from tqdm import tqdm

from .tensorboard_logger import TensorboardLogger


class DefaultTrainer:
    """ Trainer class for 2d/3d training on a single GPU.
    """
    def __init__(
        self,
        name,
        train_loader=None,
        val_loader=None,
        model=None,
        loss=None,
        optimizer=None,
        metric=None,
        device=None,
        lr_scheduler=None,
        log_image_interval=100,
        mixed_precision=True,
        early_stopping=None,
        logger=TensorboardLogger
    ):
        self.name = name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.metric = metric
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.log_image_interval = log_image_interval

        self._iteration = 0
        self._epoch = 0
        self._best_epoch = 0

        self.mixed_precision = mixed_precision
        self.early_stopping = early_stopping

        self.scaler = amp.GradScaler() if self.mixed_precision else None
        self.checkpoint_folder = os.path.join('./checkpoints', self.name)

        log_dir = os.path.join('./logs', self.name)
        self.logger = None if logger is None else logger(log_dir,
                                                         log_image_interval)

    def _initialize(self, iterations, load_from_checkpoint):
        assert self.train_loader is not None
        assert self.val_loader is not None
        assert self.model is not None
        assert self.loss is not None
        assert self.optimizer is not None
        assert self.metric is not None
        assert self.device is not None

        if load_from_checkpoint is not None:
            self.load_checkpoint(load_from_checkpoint)

        self.max_iteration = self._iteration + iterations
        epochs = int(np.ceil(float(iterations) / len(self.train_loader)))
        self.max_epoch = self._epoch + epochs

        self.model.to(self.device)
        self.loss.to(self.device)

        os.makedirs(self.checkpoint_folder, exist_ok=True)
        os.makedirs('./logs', exist_ok=True)

        print("Start fitting for", self.max_iteration - self._iteration,
              "iterations / ", self.max_epoch - self._epoch, "epochs")

        best_metric = np.inf
        return best_metric

    def save_checkpoint(self, name, best_metric):
        save_path = os.path.join(self.checkpoint_folder, f'{name}.pt')
        save_dict = {
            'iteration': self._iteration,
            'epoch': self._epoch,
            'best_epoch': self._best_epoch,
            'best_metric': best_metric,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }
        if self.scaler is not None:
            save_dict.update({'scaler_state': self.scaler.state_dict()})
        if self.lr_scheduler is not None:
            save_dict.update({'scheduler_state': self.lr_scheduler.state_dict()})
        torch.save(save_dict, save_path)

    def load_checkpoint(self, name='best'):
        save_path = os.path.join(self.checkpoint_folder, f'{name}.pt')
        if not os.path.exists(save_path):
            warnings.warn(f"Cannot load checkpoint. {save_path} does not exist.")
            return

        save_dict = torch.load(save_path)

        self._iteration = save_dict['iteration']
        self._epoch = save_dict['epoch']
        self._ebest_poch = save_dict['best_epoch']
        self.best_metric = save_dict['best_metric']

        self.model.load_state_dict(save_dict['model_state'])
        # we need to send the network to the device before loading the optimizer state!
        self.model.to(self.device)

        self.optimizer.load_state_dict(save_dict['optimizer_state'])
        if self.scaler is not None:
            self.scaler.load_state_dict(save_dict['scaler_state'])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(save_dict['scheduler_state'])

    def fit(self, iterations, load_from_checkpoint=None):
        best_metric = self._initialize(iterations, load_from_checkpoint)

        if self.mixed_precision:
            train_epoch = self._train_epoch_mixed
            validate = self._validate_mixed
            print("Training with mixed precision")
        else:
            train_epoch = self._train_epoch
            validate = self._validate
            print("Training with single precision")

        # TODO pass the progress to training and update after each iteration
        progress = tqdm(total=iterations, desc=f"Epoch {self._epoch}", leave=True)
        msg = "Epoch %i: average [s/it]: %f, current metric: %f, best metric: %f"

        train_epochs = self.max_epoch - self._epoch
        for _ in range(train_epochs):
            t_per_iter = train_epoch(progress)
            current_metric = validate()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(current_metric)

            if current_metric < best_metric:
                best_metric = current_metric
                self._best_epoch = self._epoch
                self.save_checkpoint('best', best_metric)

            # TODO for tiny epochs we don't want to save every time
            self.save_checkpoint('latest', best_metric)
            if self.early_stopping is not None:
                epochs_since_best = self._epoch - self._best_epoch
                if epochs_since_best > self.early_stopping:
                    print("Stopping training because there has been no improvement for", self.early_stopping, "epochs")
                    break

            self._epoch += 1
            progress.set_description(msg % (self._epoch, t_per_iter, current_metric, best_metric),
                                     refresh=True)

        print(f"Finished training after {self._epoch} epochs / {self._iteration} iterations.")
        print(f"The best epoch is number {self._best_epoch}.")

    def _train_epoch(self, progress):
        self.model.train()

        n_iter = 0
        t_per_iter = time.time()
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            prediction = self.model(x)
            if self._iteration % self.log_image_interval == 0:
                prediction.retain_grad()
            loss = self.loss(prediction, y)

            loss.backward()
            self.optimizer.step()

            lr = [pm['lr'] for pm in self.optimizer.param_groups][0]
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
                loss = self.loss(prediction, y)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            lr = [pm['lr'] for pm in self.optimizer.param_groups][0]
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
                loss += self.loss(prediction, y).item()
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
                    loss = self.loss(prediction, y)
                    metric = self.metric(prediction, y)
                loss_val += loss
                metric_val += metric

        metric_val /= len(self.val_loader)
        loss_val /= len(self.val_loader)
        if self.logger is not None:
            self.logger.log_validation(metric, loss,
                                       x, y, prediction)
        return metric_val
