"""@private
"""

from typing import List, Optional
import time
import torch
import torch.nn as nn
import torch_em


class DummyLoss(torch.nn.Module):
    pass


class ProbabilisticUNetTrainer(torch_em.trainer.DefaultTrainer):
    """Trainer for the Probabilistic UNet (Kohl et al., https://arxiv.org/abs/1806.05034).

    Combines a UNet encoder with VAE prior and posterior networks to produce generative
    segmentations. Training uses the ELBO loss via the posterior; validation samples from
    the prior.

    Args:
        clipping_value: Maximum gradient norm for clipping. No clipping if None.
        prior_samples: Number of prior samples drawn when logging segmentation outputs.
        loss: Loss callable for training, e.g. ProbabilisticUNetLoss. Must not be None.
        loss_and_metric: Loss and metric callable for validation. Must not be None.
    """

    def __init__(
        self,
        clipping_value: Optional[float] = None,
        prior_samples: int = 16,
        loss: Optional[nn.Module] = None,
        loss_and_metric: Optional[nn.Module] = None,
        **kwargs
    ) -> None:
        assert loss is not None and loss_and_metric is not None
        super().__init__(loss=loss, metric=DummyLoss(), **kwargs)

        self.loss_and_metric = loss_and_metric
        self.clipping_value = clipping_value
        self.prior_samples = prior_samples

    def _backprop(self, loss: torch.Tensor) -> None:
        loss.backward()
        if self.clipping_value is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipping_value)
        self.optimizer.step()

    def _backprop_mixed(self, loss: torch.Tensor) -> None:
        self.scaler.scale(loss).backward()
        if self.clipping_value is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipping_value)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def _sample(self) -> List[torch.Tensor]:
        samples = [self.model.sample() for _ in range(self.prior_samples)]
        return samples

    def _train_epoch_impl(self, progress, forward_context, backprop) -> float:
        self.model.train()

        n_iter = 0
        t_per_iter = time.time()

        for x, y in self.train_loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with forward_context():
                loss = self.loss(self.model, x, y)

            backprop(loss)

            if self.logger is not None:
                lr = [pm["lr"] for pm in self.optimizer.param_groups][0]
                samples = self._sample() if self._iteration % self.log_image_interval == 0 else None
                y = y[:, :self.model.output_channels, ...]
                self.logger.log_train(self._iteration, loss, lr, x, y, samples)

            self._iteration += 1
            n_iter += 1
            if self._iteration >= self.max_iteration:
                break
            progress.update(1)

        t_per_iter = (time.time() - t_per_iter) / n_iter
        return t_per_iter

    def _validate_impl(self, forward_context) -> float:
        self.model.eval()

        metric_val = 0.0
        loss_val = 0.0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

                with forward_context():
                    loss, metric = self.loss_and_metric(self.model, x, y)

                loss_val += loss.item()
                metric_val += metric.item()

        metric_val /= len(self.val_loader)
        loss_val /= len(self.val_loader)

        if self.logger is not None:
            samples = self._sample()
            y = y[:, :self.model.output_channels, ...]
            self.logger.log_validation(self._iteration, metric_val, loss_val, x, y, samples)

        return metric_val
