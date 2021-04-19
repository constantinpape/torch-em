import torch
import torch.nn as nn
from . import contrastive_impl as impl


def check_consecutive(labels):
    """ Check that the input labels are consecutive and start at zero.
    """
    diff = labels[1:] - labels[:-1]
    return (labels[0] == 0) and (diff == 1).all()


# TODO add less memory hungry (CUDA?) implementation
# TODO support ignore labels:
# - ignore_label: ignored in all terms
# - ignore_dist: ignored in distance term
# - ignore_var: ignored in variance term
class ContrastiveLoss(nn.Module):
    """ Implementation of contrastive loss defined in https://arxiv.org/pdf/1708.02551.pdf
    'Semantic Instance Segmentation with a Discriminative Loss Function'
    """
    def __init__(self, delta_var, delta_dist, norm='fro', alpha=1., beta=1., gamma=0.001):
        super().__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    # This implementation expands all tensors to match the instance dimensions.
    # Hence it's fast, but has high memory consumption.
    # The implementation does not support masking any instance labels in the loss.
    def _expand_impl_batch(self, input_batch, target_batch, n_dim):
        # add singleton batch dimension required for further computation
        input_batch = input_batch.unsqueeze(0)

        # get number of instances in the batch
        instances = torch.unique(target_batch)
        assert check_consecutive(instances)
        n_instances = instances.size()[0]

        # SPATIAL = D X H X W in 3d case, H X W in 2d case
        # expand each label as a one-hot vector: N x SPATIAL -> N x C x SPATIAL
        target_batch = impl.expand_as_one_hot(target_batch, n_instances)

        cluster_means, embeddings_per_instance = impl._compute_cluster_means(input_batch,
                                                                             target_batch, n_dim)
        variance_term = impl._compute_variance_term(cluster_means, embeddings_per_instance,
                                                    target_batch, n_dim, self.norm, self.delta_var)
        distance_term = impl._compute_distance_term(cluster_means, n_instances, n_dim, self.norm, self.delta_dist)
        regularization_term = impl._compute_regularizer_term(cluster_means, n_instances, n_dim, self.norm)

        # compute total loss
        return self.alpha * variance_term + self.beta * distance_term + self.gamma * regularization_term

    def forward(self, input_, target):
        n_batches = input_.shape[0]
        assert target.dim() == input_.dim()
        assert target.shape[1] == 1
        assert n_batches == target.shape[0]
        assert input_.size()[2:] == target.size()[2:]

        n_dim = input_.dim() - 2
        assert n_dim in (2, 3)

        # iterate over the batches
        loss = 0.
        for input_batch, target_batch in zip(input_, target):
            loss_batch = self._expand_impl_batch(input_batch, target_batch, n_dim)
            loss += loss_batch

        return loss.div(n_batches)
