"""@private
"""

import time
import torch
import torch_em


class DummyLoss(torch.nn.Module):
    pass


class ProbabilisticUNetTrainer(torch_em.trainer.DefaultTrainer):
    """This trainer implements training for the 'Probabilistic UNet' of Kohl et al.: (https://arxiv.org/abs/1806.05034).
    This approach combines the learnings from UNet and VAEs (Prior and Posterior networks) to obtain generative
    segmentations. The heuristic trains by taking into account the feature maps from UNet and the samples from
    the posterior distribution, estimating the loss and further sampling from the prior for validation.

    Args:
        clipping_value [float] - (default: None)
        prior_samples [int] - (default: 16)
        loss [callable] - (default: None)
        loss_and_metric [callable] - (default: None)
    """

    def __init__(
            self,
            clipping_value=None,
            prior_samples=16,
            loss=None,
            loss_and_metric=None,
            **kwargs
    ):
        super().__init__(loss=loss, metric=DummyLoss(), **kwargs)
        assert loss, loss_and_metric is not None

        self.loss_and_metric = loss_and_metric

        self.clipping_value = clipping_value

        self.prior_samples = prior_samples
        self.sigmoid = torch.nn.Sigmoid()

        self._kwargs = kwargs

    #
    # functionality for sampling from the network
    #

    def _sample(self):
        samples = [self.model.sample() for _ in range(self.prior_samples)]
        return samples

    #
    # training and validation functionality
    #

    def _train_epoch_impl(self, progress, forward_context, backprop):
        self.model.train()

        n_iter = 0
        t_per_iter = time.time()

        for x, y in self.train_loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with forward_context():
                # We pass the model, the input and the labels to the supervised loss function, so
                # that's how the loss is calculated stays flexible, e.g. here to enable ELBO for PUNet.
                loss = self.loss(self.model, x, y)

            backprop(loss)

            # To counter the exploding gradients in the posterior net
            if self.clipping_value is not None:
                torch.nn.utils.clip_grad_norm_(self.model.posterior.encoder.layers.parameters(), self.clipping_value)

            if self.logger is not None:
                lr = [pm["lr"] for pm in self.optimizer.param_groups][0]
                samples = self._sample() if self._iteration % self.log_image_interval == 0 else None
                self.logger.log_train(self._iteration, loss, lr, x, y, samples)

            self._iteration += 1
            n_iter += 1
            if self._iteration >= self.max_iteration:
                break
            progress.update(1)

        t_per_iter = (time.time() - t_per_iter) / n_iter
        return t_per_iter

    def _validate_impl(self, forward_context):
        self.model.eval()

        metric_val = 0.0
        loss_val = 0.0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

                with forward_context():
                    loss, metric = self.loss_and_metric(self.model, x, y)

                loss_val += loss.item()
                metric_val += metric

        metric_val /= len(self.val_loader)
        loss_val /= len(self.val_loader)

        if self.logger is not None:
            samples = self._sample()
            self.logger.log_validation(self._iteration, metric_val, loss_val, x, y, samples)

        return metric_val
