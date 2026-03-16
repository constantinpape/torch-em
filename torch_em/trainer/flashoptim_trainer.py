import time

import torch

try:
    from flashoptim import FlashAdamW, cast_model
except ImportError:
    FlashAdamW = None
    cast_model = None

from .default_trainer import DefaultTrainer


class FlashOptimTrainer(DefaultTrainer):
    """Trainer for training models with FlashOptim optimizers for memory-efficiency.

    The trainer adapts the `DefaultTrainer` for the following reasons:
    1. Casts the model parameters and input data to bf16 precision (see `torch.bfloat16` for details)
    2. Sets `mixed_precision` and `compile_model` to `False`.

    NOTE: There are a couple of things to keep in mind:
    1. Multi-GPU training (eg. using DDP) is currently not supported.
    2. Gradient clipping cannot be applied to the parameters.
    3. Gradient scaling (eg. using `torch.amp.GradScaler`) is currently not supported.
    4. Microbatch accumulation (gradient accumulation) is not possible.

    For details, check out the official repository: https://github.com/databricks/flashoptim.
    And please cite https://doi.org/10.48550/arXiv.2602.23349 if you use this trainer for your research.
    """
    def __init__(self, **kwargs):
        if FlashAdamW is None:
            raise ImportError(
                "flashoptim is required for `FlashOptimTrainer`. Please install it using `pip install flashoptim`."
            )

        # Pinning the values for 'mixed_precision' and 'compile_model' both to 'False'.
        kwargs["mixed_precision"] = False
        kwargs["compile_model"] = False

        super().__init__(**kwargs)
        self._kwargs = {}  # Required by the serializer.

        # This function casts the model parameters to bf16.
        cast_model(self.model, dtype=torch.bfloat16)

        lr = self.optimizer.param_groups[0]["lr"]

        # Choice of FlashOptim optimizer - a direct drop-in replacement to PyTorch optimizer API.
        self.optimizer = FlashAdamW(self.model.parameters(), lr=lr)

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

    def _train_epoch_impl(self, progress, forward_context, backprop):
        self.model.train()

        n_iter = 0
        t_per_iter = time.time()
        for x, y in self.train_loader:
            # Casts inputs to bf16 precision.
            x = x.to(self.device, non_blocking=True).to(torch.bfloat16)
            y = y.to(self.device, non_blocking=True).to(torch.bfloat16)

            self.optimizer.zero_grad()

            with forward_context():
                pred, loss = self._forward_and_loss(x, y)

            backprop(loss)

            lr = [pm["lr"] for pm in self.optimizer.param_groups][0]
            if self.logger is not None:
                self.logger.log_train(self._iteration, loss, lr, x, y, pred, log_gradients=True)

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
                # Casts inputs to bf16 precision.
                x = x.to(self.device, non_blocking=True).to(torch.bfloat16)
                y = y.to(self.device, non_blocking=True).to(torch.bfloat16)

                with forward_context():
                    pred, loss = self._forward_and_loss(x, y)
                    metric = self.metric(pred, y)

                loss_val += loss.item()
                metric_val += metric.item()

        metric_val /= len(self.val_loader)
        loss_val /= len(self.val_loader)
        if self.logger is not None:
            self.logger.log_validation(self._iteration, metric_val, loss_val, x, y, pred)
        return metric_val
