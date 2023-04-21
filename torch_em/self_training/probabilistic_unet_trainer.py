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

    Parameters:
        clip_posterior [bool] - (default: False)
        prior_samples [int] - (default: 16)
        supervised_loss [callable] - (default: None)
        supervised_loss_and_metric [callable] - (default: None)
    """

    def __init__(
            self,
            clip_posterior=False,
            prior_samples=16,
            supervised_loss=None,
            supervised_loss_and_metric=None,
            **kwargs
    ):
        super().__init__(**kwargs)
        assert supervised_loss, supervised_loss_and_metric is not None

        self.supervised_loss = supervised_loss
        self.supervised_loss_and_metric = supervised_loss_and_metric

        self.clip_posterior = clip_posterior
        self.clipping_value = 1

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
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            with forward_context():
                # We pass the model, the input and the labels to the supervised loss function, so
                # that's how the loss is calculated stays flexible, e.g. here to enable ELBO for PUNet.
                loss = self.supervised_loss(self.model, x, y)

            backprop(loss)

            # To counter the exploding gradients in the posterior net
            if self.clip_posterior:
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
                x, y = x.to(self.device), y.to(self.device)

                with forward_context():
                    loss, metric = self.supervised_loss_and_metric(self.model, x, y)

                loss_val += loss.item()
                metric_val += metric

        metric_val /= len(self.val_loader)
        loss_val /= len(self.val_loader)

        if self.logger is not None:
            samples = self._sample()
            self.logger.log_validation(self._iteration, metric_val, loss_val, x, y, samples)

        return metric_val
