from typing import List

import torch
import torch.nn as nn


__all__ = ["BatchRenorm2d", "BatchRenorm3d"]


class BatchRenormNd(nn.Module):
    """Base class for BatchRenorm Module: https://arxiv.org/abs/1702.03275

    Inspired by https://pytorch.org/rl/main/_modules/torchrl/modules/models/batchrenorm.html#BatchRenorm1d.

    And the original code is adapted from https://github.com/google-research/corenet.
    """

    def __init__(
        self,
        num_features: int,
        *,
        dims: List[int],
        momentum: float = 0.01,
        eps: float = 1e-5,
        max_r: float = 3.0,
        max_d: float = 5.0,
        warmup_steps: int = 10000,
        smooth: bool = False,
    ):
        super().__init__()

        self.num_features = num_features
        self.dims = dims
        self.eps = eps
        self.momentum = momentum
        self.max_r = max_r
        self.max_d = max_d
        self.warmup_steps = warmup_steps
        self.smooth = smooth

        self.register_buffer("running_mean", torch.zeros(num_features, dtype=torch.float32))
        self.register_buffer("running_var", torch.ones(num_features, dtype=torch.float32))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.int64))

        self.weight = nn.Parameter(torch.ones(num_features, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(num_features, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.dim() >= 2:
            raise ValueError(
                f"The {type(self).__name__} expects a 2D (or more) dimensional tensor, got {x.dim()}D."
            )

        view_dims = [1, x.shape[1]] + [1] * (x.dim() - 2)

        def _v(v):
            return v.view(view_dims)

        running_std = torch.sqrt(self.running_var + self.eps)

        if self.training:
            b_mean = x.mean(dim=self.dims)
            b_var = x.var(dim=self.dims, unbiased=False)
            b_std = (b_var + self.eps).sqrt_()

            r = torch.clamp((b_std.detach() / running_std), 1 / self.max_r, self.max_r)
            d = torch.clamp(
                (b_mean.detach() - self.running_mean) / running_std,
                -self.max_d,
                self.max_d,
            )

            # Warmup factor
            if self.warmup_steps > 0:
                if self.smooth:
                    warmup_factor = self.num_batches_tracked / self.warmup_steps
                else:
                    warmup_factor = self.num_batches_tracked // self.warmup_steps
                warmup_factor = torch.clamp(warmup_factor, 0.0, 1.0)
                r = 1.0 + (r - 1.0) * warmup_factor
                d = d * warmup_factor

            x = (x - _v(b_mean)) / _v(b_std) * _v(r) + _v(d)

            num_elements = x.numel() // x.shape[1]
            unbiased_var = b_var.detach() * num_elements / max(num_elements - 1, 1)

            self.running_var += self.momentum * (unbiased_var - self.running_var)
            self.running_mean += self.momentum * (b_mean.detach() - self.running_mean)
            self.num_batches_tracked += 1
            self.num_batches_tracked.clamp_max_(self.warmup_steps)
        else:
            x = (x - _v(self.running_mean)) / _v(running_std)

        return _v(self.weight) * x + _v(self.bias)


class BatchRenorm2d(BatchRenormNd):
    """BatchRenorm2d Module (https://arxiv.org/abs/1702.03275).

    Adapted for 2D feature maps. Use with input of shape [B, C, H, W].

    Args:
        num_features: Number of input feature channels.
        momentum: Momentum factor for computing the running mean and variance. Defaults to ``0.01``.
        eps: Small value added to the variance to avoid division by zero. Defaults to ``1e-5``.
        max_r: Maximum value for the scaling factor r. Defaults to ``3.0``.
        max_d: Maximum value for the bias factor d. Defaults to ``5.0``.
        warmup_steps: Number of warm-up steps for the running mean and variance. Defaults to ``10000``.
        smooth: if ``True``, the behavior smoothly transitions from regular batch-norm (when ``iter=0``)
            to batch-renorm (when ``iter=warmup_steps``). Otherwise, the behavior will transition from
            batch-norm to batch-renorm when ``iter=warmup_steps``. Defaults to ``False``.
    """

    def __init__(self, num_features: int, **kwargs):
        super().__init__(num_features, dims=[0, 2, 3], **kwargs)


class BatchRenorm3d(BatchRenormNd):
    """BatchRenorm3d Module (https://arxiv.org/abs/1702.03275).

    Adapted for 3D volumetric data. Use with input of shape [B, C, D, H, W].

    Args:
        num_features: Number of input feature channels.
        momentum: Momentum factor for computing the running mean and variance. Defaults to ``0.01``.
        eps: Small value added to the variance to avoid division by zero. Defaults to ``1e-5``.
        max_r: Maximum value for the scaling factor r. Defaults to ``3.0``.
        max_d: Maximum value for the bias factor d. Defaults to ``5.0``.
        warmup_steps: Number of warm-up steps for the running mean and variance. Defaults to ``10000``.
        smooth: if ``True``, the behavior smoothly transitions from regular batch-norm (when ``iter=0``)
            to batch-renorm (when ``iter=warmup_steps``). Otherwise, the behavior will transition from
            batch-norm to batch-renorm when ``iter=warmup_steps``. Defaults to ``False``.
    """

    def __init__(self, num_features: int, **kwargs):
        super().__init__(num_features, dims=[0, 2, 3, 4], **kwargs)
