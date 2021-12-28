from warnings import warn

import torch
import torch.nn as nn
from . import contrastive_impl as impl


def check_consecutive(labels):
    """Check that the input labels are consecutive and start at zero.
    """
    diff = labels[1:] - labels[:-1]
    return (labels[0] == 0) and (diff == 1).all()


# TODO support more sophisticated ignore labels:
# - ignore_dist: ignored in distance term
# - ignore_var: ignored in variance term
class ContrastiveLoss(nn.Module):
    """Implementation of contrastive loss defined in https://arxiv.org/pdf/1708.02551.pdf
    Semantic Instance Segmentation with a Discriminative Loss Function

    This class contians different implementations for the discrimnative loss:
    - based on pure pytorch, expanding the instance dimension, this is not memory efficient
    - based on pytorch_scatter (https://github.com/rusty1s/pytorch_scatter), this is memory efficient

    Arguments:
        delta_var [float] -
        delta_dist [float] -
        norm [str] -
        aplpha [float] -
        beta [float] -
        gamma [float] -
        ignore_label [int] -
        impl [str] -
    """
    implementations = (None, "scatter", "expand")

    def __init__(self, delta_var, delta_dist, norm="fro",
                 alpha=1., beta=1., gamma=0.001,
                 ignore_label=None, impl=None):
        assert ignore_label is None, "Not implemented"  # TODO
        super().__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ignore_label = ignore_label

        assert impl in self.implementations
        has_torch_scatter = self.has_torch_scatter()
        if impl is None:
            if not has_torch_scatter:
                pt_scatter = "https://github.com/rusty1s/pytorch_scatter"
                warn(f"ContrastiveLoss: using pure pytorch implementation. Install {pt_scatter} for memory efficiency.")
            self._contrastive_impl = self._scatter_impl_batch if has_torch_scatter else self._expand_impl_batch
        elif impl == "scatter":
            assert has_torch_scatter
            self._contrastive_impl = self._scatter_impl_batch
        elif impl == "expand":
            self._contrastive_impl = self._expand_impl_batch

        # all torch_em classes should store init kwargs to easily recreate the init call
        self.init_kwargs = {"delta_var": delta_var, "delta_dist": delta_dist, "norm": norm,
                            "alpha": alpha, "beta": beta, "gamma": gamma, "ignore_label": ignore_label,
                            "impl": impl}

    @staticmethod
    def has_torch_scatter():
        try:
            import torch_scatter
        except ImportError:
            torch_scatter = None
        return torch_scatter is not None

    # This implementation expands all tensors to match the instance dimensions.
    # Hence it's fast, but has high memory consumption.
    # The implementation does not support masking any instance labels in the loss.
    def _expand_impl_batch(self, input_batch, target_batch, ndim):
        # add singleton batch dimension required for further computation
        input_batch = input_batch.unsqueeze(0)

        # get number of instances in the batch
        instances = torch.unique(target_batch)
        assert check_consecutive(instances), f"{instances}"
        n_instances = instances.size()[0]

        # SPATIAL = D X H X W in 3d case, H X W in 2d case
        # expand each label as a one-hot vector: N x SPATIAL -> N x C x SPATIAL
        target_batch = impl.expand_as_one_hot(target_batch, n_instances)

        cluster_means, embeddings_per_instance = impl._compute_cluster_means(input_batch,
                                                                             target_batch, ndim)
        variance_term = impl._compute_variance_term(cluster_means, embeddings_per_instance,
                                                    target_batch, ndim, self.norm, self.delta_var)
        distance_term = impl._compute_distance_term(cluster_means, n_instances,
                                                    ndim, self.norm, self.delta_dist)
        regularization_term = impl._compute_regularizer_term(cluster_means, n_instances,
                                                             ndim, self.norm)
        # compute total loss
        return self.alpha * variance_term + self.beta * distance_term + self.gamma * regularization_term

    def _scatter_impl_batch(self, input_batch, target_batch, ndim):
        # add singleton batch dimension required for further computation
        input_batch = input_batch.unsqueeze(0)

        instance_ids, instance_sizes = torch.unique(target_batch, return_counts=True)
        n_instances = len(instance_ids)
        cluster_means = impl._compute_cluster_means_scatter(input_batch, target_batch, ndim,
                                                            n_lbl=n_instances)

        variance_term = impl._compute_variance_term_scatter(cluster_means, input_batch, target_batch, self.norm,
                                                            self.delta_var, instance_sizes)
        distance_term = impl._compute_distance_term_scatter(cluster_means, self.norm, self.delta_dist)

        regularization_term = torch.sum(
            torch.norm(cluster_means, p=self.norm, dim=1)
        ).div(n_instances)

        # compute total loss
        return self.alpha * variance_term + self.beta * distance_term + self.gamma * regularization_term

    def forward(self, input_, target):
        n_batches = input_.shape[0]
        assert target.dim() == input_.dim()
        assert target.shape[1] == 1
        assert n_batches == target.shape[0]
        assert input_.size()[2:] == target.size()[2:]

        ndim = input_.dim() - 2
        assert ndim in (2, 3)

        # iterate over the batches
        loss = 0.0
        for input_batch, target_batch in zip(input_, target):
            loss_batch = self._contrastive_impl(input_batch, target_batch, ndim)
            loss += loss_batch

        return loss.div(n_batches)
