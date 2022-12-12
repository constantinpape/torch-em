import math

import numpy as np
import torch
import torch.nn as nn
try:
    from torch_scatter import scatter_mean
except ImportError:
    scatter_mean = None

from . import contrastive_impl as cimpl
from .affinity_side_loss import AffinitySideLoss
from .dice import DiceLoss


def compute_cluster_means(embeddings, target, n_instances):
    """
    Computes mean embeddings per instance.
    E - embedding dimension

    Args:
        embeddings: tensor of pixel embeddings, shape: ExSPATIAL
        target: one-hot encoded target instances, shape: SPATIAL
        n_instances: number of instances
    """
    assert scatter_mean is not None, "torch_scatter is required"
    embeddings = embeddings.flatten(1)
    target = target.flatten()
    mean_embeddings = scatter_mean(embeddings, target, dim_size=n_instances)
    return mean_embeddings.transpose(1, 0)


def select_stable_anchor(embeddings, mean_embedding, object_mask, delta_var, norm="fro"):
    """
    Anchor sampling procedure. Given a binary mask of an object (`object_mask`) and a `mean_embedding` vector within
    the mask, the function selects a pixel from the mask at random and returns its embedding only if it"s closer than
    `delta_var` from the `mean_embedding`.

    Args:
        embeddings (torch.Tensor): ExSpatial vector field of an image
        mean_embedding (torch.Tensor): E-dimensional mean of embeddings lying within the `object_mask`
        object_mask (torch.Tensor): binary image of a selected object
        delta_var (float): contrastive loss, pull force margin
        norm (str): vector norm used, default: Frobenius norm

    Returns:
        embedding of a selected pixel within the mask or the mean embedding if stable anchor could be found
    """
    indices = torch.nonzero(object_mask, as_tuple=True)
    # convert to numpy
    indices = [t.cpu().numpy() for t in indices]

    # randomize coordinates
    seed = np.random.randint(np.iinfo("int32").max)
    for t in indices:
        rs = np.random.RandomState(seed)
        rs.shuffle(t)

    for ind in range(len(indices[0])):
        if object_mask.dim() == 2:
            y, x = indices
            anchor_emb = embeddings[:, y[ind], x[ind]]
            anchor_emb = anchor_emb[..., None, None]
        else:
            z, y, x = indices
            anchor_emb = embeddings[:, z[ind], y[ind], x[ind]]
            anchor_emb = anchor_emb[..., None, None, None]
        dist_to_mean = torch.norm(mean_embedding - anchor_emb, norm)
        if dist_to_mean < delta_var:
            return anchor_emb
    # if stable anchor has not been found, return mean_embedding
    return mean_embedding


class GaussianKernel(nn.Module):
    def __init__(self, delta_var, pmaps_threshold):
        super().__init__()
        self.delta_var = delta_var
        # dist_var^2 = -2*sigma*ln(pmaps_threshold)
        self.two_sigma = delta_var * delta_var / (-math.log(pmaps_threshold))

    def forward(self, dist_map):
        return torch.exp(- dist_map * dist_map / self.two_sigma)


class CombinedAuxLoss(nn.Module):
    def __init__(self, losses, weights):
        super().__init__()
        self.losses = losses
        self.weights = weights

    def forward(self, embeddings, target, instance_pmaps, instance_masks):
        result = 0.
        for loss, weight in zip(self.losses, self.weights):
            if isinstance(loss, AffinitySideLoss):
                # add batch axis / batch and channel axis for embeddings, target
                result += weight * loss(embeddings[None], target[None, None])
            elif instance_masks is not None:
                result += weight * loss(instance_pmaps, instance_masks).mean()
        return result


class ContrastiveLossBase(nn.Module):
    """Base class for the spoco losses.
    """

    def __init__(self, delta_var, delta_dist,
                 norm="fro", alpha=1., beta=1., gamma=0.001, unlabeled_push_weight=0.0,
                 instance_term_weight=1.0, impl=None):
        assert scatter_mean is not None, "Spoco loss requires pytorch_scatter"
        super().__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.unlabeled_push_weight = unlabeled_push_weight
        self.unlabeled_push = unlabeled_push_weight > 0
        self.instance_term_weight = instance_term_weight

    def __str__(self):
        return super().__str__() + f"\ndelta_var: {self.delta_var}\ndelta_dist: {self.delta_dist}" \
                                   f"\nalpha: {self.alpha}\nbeta: {self.beta}\ngamma: {self.gamma}" \
                                   f"\nunlabeled_push_weight: {self.unlabeled_push_weight}" \
                                   f"\ninstance_term_weight: {self.instance_term_weight}"

    def _compute_variance_term(self, cluster_means, embeddings, target, instance_counts, ignore_zero_label):
        """Computes the variance term, i.e. intra-cluster pull force that draws embeddings towards the mean embedding

        C - number of clusters (instances)
        E - embedding dimension
        SPATIAL - volume shape, i.e. DxHxW for 3D/ HxW for 2D

        Args:
            cluster_means: mean embedding of each instance, tensor (CxE)
            embeddings: embeddings vectors per instance, tensor (ExSPATIAL)
            target: label tensor (1xSPATIAL); each label is represented as one-hot vector
            instance_counts: number of voxels per instance
            ignore_zero_label: if True ignores the cluster corresponding to the 0-label
        """
        assert target.dim() in (2, 3)
        ignore_labels = [0] if ignore_zero_label else None
        return cimpl._compute_variance_term_scatter(
            cluster_means, embeddings.unsqueeze(0), target.unsqueeze(0),
            self.norm, self.delta_var, instance_counts, ignore_labels
        )

    def _compute_unlabeled_push(self, cluster_means, embeddings, target):
        assert target.dim() in (2, 3)
        n_instances = cluster_means.shape[0]

        # permute embedding dimension at the end
        if target.dim() == 2:
            embeddings = embeddings.permute(1, 2, 0)
        else:
            embeddings = embeddings.permute(1, 2, 3, 0)

        # decrease number of instances `C` since we're ignoring 0-label
        n_instances -= 1
        # if there is only 0-label in the target return 0
        if n_instances == 0:
            return 0.0

        background_mask = target == 0
        n_background = background_mask.sum()
        background_push = 0.0
        # skip embedding corresponding to the background pixels
        for cluster_mean in cluster_means[1:]:
            # compute distances between embeddings and a given cluster_mean
            dist_to_mean = torch.norm(embeddings - cluster_mean, self.norm, dim=-1)
            # apply background mask and compute hinge
            dist_hinged = torch.clamp((self.delta_dist - dist_to_mean) * background_mask, min=0) ** 2
            background_push += torch.sum(dist_hinged) / n_background

        # normalize by the number of instances
        return background_push / n_instances

    # def _compute_distance_term_scatter(cluster_means, norm, delta_dist):
    def _compute_distance_term(self, cluster_means, ignore_zero_label):
        """
        Compute the distance term, i.e an inter-cluster push-force that pushes clusters away from each other, increasing
        the distance between cluster centers

        Args:
            cluster_means: mean embedding of each instance, tensor (CxE)
            ignore_zero_label: if True ignores the cluster corresponding to the 0-label
        """
        ignore_labels = [0] if ignore_zero_label else None
        return cimpl._compute_distance_term_scatter(cluster_means, self.norm, self.delta_dist, ignore_labels)

    def _compute_regularizer_term(self, cluster_means):
        """
        Computes the regularizer term, i.e. a small pull-force that draws all clusters towards origin to keep
        the network activations bounded
        """
        # compute the norm of the mean embeddings
        norms = torch.norm(cluster_means, p=self.norm, dim=1)
        # return the average norm per batch
        return torch.sum(norms) / cluster_means.size(0)

    def compute_instance_term(self, embeddings, cluster_means, target):
        """Computes auxiliary loss based on embeddings and a given list of target
        instances together with their mean embeddings

        Args:
            embeddings (torch.tensor): pixel embeddings (ExSPATIAL)
            cluster_means (torch.tensor): mean embeddings per instance (CxExSINGLETON_SPATIAL)
            target (torch.tensor): ground truth instance segmentation (SPATIAL)

        Returns:
            float: value of the instance-based term
        """
        raise NotImplementedError

    def forward(self, input_, target):
        """
        Args:
             input_ (torch.tensor): embeddings predicted by the network (NxExDxHxW) (E - embedding dims)
                expects float32 tensor
             target (torch.tensor): ground truth instance segmentation (Nx1DxHxW)
                expects int64 tensor
        Returns:
            Combined loss defined as: alpha * variance_term + beta * distance_term + gamma * regularization_term
                + instance_term_weight * instance_term + unlabeled_push_weight * unlabeled_push_term
        """
        # enable calling this loss from the spoco trainer, which passes a tuple
        if isinstance(input_, tuple):
            assert len(input_) == 2
            input_ = input_[0]

        n_batches = input_.shape[0]
        # compute the loss per each instance in the batch separately
        # and sum it up in the per_instance variable
        loss = 0.0
        for single_input, single_target in zip(input_, target):
            # compare spatial dimensions
            assert single_input.shape[1:] == single_target.shape[1:], f"{single_input.shape}, {single_target.shape}"
            assert single_target.shape[0] == 1
            single_target = single_target[0]

            contains_bg = 0 in single_target
            ignore_zero_label = self.unlabeled_push and contains_bg

            # get number of instances in the batch instance
            instance_ids, instance_counts = torch.unique(single_target, return_counts=True)

            # get the number of instances
            C = instance_ids.size(0)

            # compute mean embeddings (output is of shape CxE)
            cluster_means = compute_cluster_means(single_input, single_target, C)

            # compute variance term, i.e. pull force
            variance_term = self._compute_variance_term(
                cluster_means, single_input, single_target, instance_counts, ignore_zero_label
            )

            # compute unlabeled push force, i.e. push force between
            # the mean cluster embeddings and embeddings of background pixels
            # compute only ignore_zero_label is True, i.e. a given patch contains background label
            unlabeled_push_term = 0.0
            if self.unlabeled_push and contains_bg:
                unlabeled_push_term = self._compute_unlabeled_push(cluster_means, single_input, single_target)

            # compute the instance-based auxiliary loss
            instance_term = self.compute_instance_term(single_input, cluster_means, single_target)

            # compute distance term, i.e. push force
            distance_term = self._compute_distance_term(cluster_means, ignore_zero_label)

            # compute regularization term
            regularization_term = self._compute_regularizer_term(cluster_means)

            # compute total loss and sum it up
            loss = self.alpha * variance_term + \
                self.beta * distance_term + \
                self.gamma * regularization_term + \
                self.instance_term_weight * instance_term + \
                self.unlabeled_push_weight * unlabeled_push_term

            loss += loss

        # reduce across the batch dimension
        return loss.div(n_batches)


class ExtendedContrastiveLoss(ContrastiveLossBase):
    """Contrastive loss extended with instance-based loss term and background push term.

    Based on:
    "Sparse Object-level Supervision for Instance Segmentation with Pixel Embeddings": https://arxiv.org/abs/2103.14572
    """

    def __init__(self, delta_var, delta_dist, norm="fro", alpha=1.0, beta=1.0, gamma=0.001,
                 unlabeled_push_weight=1.0, instance_term_weight=1.0, aux_loss="dice", pmaps_threshold=0.9, **kwargs):

        super().__init__(delta_var, delta_dist, norm=norm, alpha=alpha, beta=beta, gamma=gamma,
                         unlabeled_push_weight=unlabeled_push_weight,
                         instance_term_weight=instance_term_weight)

        # init auxiliary loss
        assert aux_loss in ["dice", "affinity", "dice_aff"]
        if aux_loss == "dice":
            self.aff_loss = None
            self.dice_loss = DiceLoss()
        # additional auxiliary losses
        elif aux_loss == "affinity":
            self.aff_loss = AffinitySideLoss(
                delta=delta_dist,
                offset_ranges=kwargs.get("offset_ranges", [(-18, 18), (-18, 18)]),
                n_samples=kwargs.get("n_samples", 9)
            )
            self.dice_loss = None
        elif aux_loss == "dice_aff":
            # combine dice and affinity side loss
            self.dice_weight = kwargs.get("dice_weight", 1.0)
            self.aff_weight = kwargs.get("aff_weight", 1.0)

            self.aff_loss = AffinitySideLoss(
                delta=delta_dist,
                offset_ranges=kwargs.get("offset_ranges", [(-18, 18), (-18, 18)]),
                n_samples=kwargs.get("n_samples", 9)
            )
            self.dice_loss = DiceLoss()

        # init dist_to_mask kernel which maps distance to the cluster center to instance probability map
        self.dist_to_mask = GaussianKernel(delta_var=self.delta_var, pmaps_threshold=pmaps_threshold)
        self.init_kwargs = {
            "delta_var": delta_var, "delta_dist": delta_dist, "norm": norm, "alpha": alpha, "beta": beta,
            "gamma": gamma, "unlabeled_push_weight": unlabeled_push_weight,
            "instance_term_weight": instance_term_weight, "aux_loss": aux_loss, "pmaps_threshold": pmaps_threshold
        }
        self.init_kwargs.update(kwargs)

    # FIXME stacking per instance here makes this very memory hungry,
    def _create_instance_pmaps_and_masks(self, embeddings, anchors, target):
        inst_pmaps = []
        inst_masks = []

        if not inst_masks:
            return None, None

        # stack along batch dimension
        inst_pmaps = torch.stack(inst_pmaps)
        inst_masks = torch.stack(inst_masks)

        return inst_pmaps, inst_masks

    def compute_instance_term(self, embeddings, cluster_means, target):
        assert embeddings.size()[1:] == target.size()

        if self.aff_loss is None:
            aff_loss = None
        else:
            aff_loss = self.aff_loss(embeddings[None], target[None, None])

        if self.dice_loss is None:
            dice_loss = None
        else:
            dice_loss = []

            # permute embedding dimension at the end
            if target.dim() == 2:
                embeddings = embeddings.permute(1, 2, 0)
            else:
                embeddings = embeddings.permute(1, 2, 3, 0)

            # compute random anchors per instance
            instances = torch.unique(target)
            for i in instances:
                if i == 0:
                    continue
                anchor_emb = cluster_means[i]
                # FIXME this makes training extremely slow, check with Adrian if this is the latest version
                # anchor_emb = select_stable_anchor(embeddings, cluster_means[i], target == i, self.delta_var)

                distance_map = torch.norm(embeddings - anchor_emb, self.norm, dim=-1)
                instance_pmap = self.dist_to_mask(distance_map).unsqueeze(0)
                instance_mask = (target == i).float().unsqueeze(0)

                dice_loss.append(self.dice_loss(instance_pmap, instance_mask))

            dice_loss = torch.tensor(dice_loss).to(embeddings.device).mean() if dice_loss else 0.0

        assert not (dice_loss is None and aff_loss is None)
        if dice_loss is None and aff_loss is not None:
            return aff_loss
        if dice_loss is not None and aff_loss is None:
            return dice_loss
        else:
            return self.dice_weight * dice_loss + self.aff_weight * aff_loss


class SPOCOLoss(ExtendedContrastiveLoss):
    """The full SPOCO Loss for instance segmentation training with sparse instance labels.

    Extends the "classic" contrastive loss with an instance-based term and a embedding consistency term.
    (The unlabeled push term is turned off by default, since we assume sparse instance labels).

    Based on:
    "Sparse Object-level Supervision for Instance Segmentation with Pixel Embeddings": https://arxiv.org/abs/2103.14572
    """

    def __init__(self, delta_var, delta_dist, norm="fro", alpha=1.0, beta=1.0, gamma=0.001,
                 unlabeled_push_weight=0.0, instance_term_weight=1.0, consistency_term_weight=1.0,
                 aux_loss="dice", pmaps_threshold=0.9, max_anchors=20, volume_threshold=0.05, **kwargs):

        super().__init__(delta_var, delta_dist, norm=norm, alpha=alpha, beta=beta, gamma=gamma,
                         unlabeled_push_weight=unlabeled_push_weight,
                         instance_term_weight=instance_term_weight,
                         aux_loss=aux_loss,
                         pmaps_threshold=pmaps_threshold,
                         **kwargs)

        self.consistency_term_weight = consistency_term_weight
        self.max_anchors = max_anchors
        self.volume_threshold = volume_threshold
        self.consistency_loss = DiceLoss()
        self.init_kwargs = {
            "delta_var": delta_var, "delta_dist": delta_dist, "norm": norm, "alpha": alpha, "beta": beta,
            "gamma": gamma, "unlabeled_push_weight": unlabeled_push_weight,
            "instance_term_weight": instance_term_weight, "aux_loss": aux_loss, "pmaps_threshold": pmaps_threshold,
            "max_anchors": max_anchors, "volume_threshold": volume_threshold
        }
        self.init_kwargs.update(kwargs)

    def __str__(self):
        return super().__str__() + f"\nconsistency_term_weight: {self.consistency_term_weight}"

    def _inst_pmap(self, emb, anchor):
        # compute distance map
        distance_map = torch.norm(emb - anchor, self.norm, dim=-1)
        # convert distance map to instance pmaps and return
        return self.dist_to_mask(distance_map)

    def emb_consistency(self, emb_q, emb_k, mask):
        inst_q = []
        inst_k = []
        for i in range(self.max_anchors):
            if mask.sum() < self.volume_threshold * mask.numel():
                break

            # get random anchor
            indices = torch.nonzero(mask, as_tuple=True)
            ind = np.random.randint(len(indices[0]))

            q_pmap = self._extract_pmap(emb_q, mask, indices, ind)
            inst_q.append(q_pmap)

            k_pmap = self._extract_pmap(emb_k, mask, indices, ind)
            inst_k.append(k_pmap)

        # stack along channel dim
        inst_q = torch.stack(inst_q)
        inst_k = torch.stack(inst_k)

        loss = self.consistency_loss(inst_q, inst_k)
        return loss

    def _extract_pmap(self, emb, mask, indices, ind):
        if mask.dim() == 2:
            y, x = indices
            anchor = emb[:, y[ind], x[ind]]
            emb = emb.permute(1, 2, 0)
        else:
            z, y, x = indices
            anchor = emb[:, z[ind], y[ind], x[ind]]
            emb = emb.permute(1, 2, 3, 0)

        return self._inst_pmap(emb, anchor)

    def forward(self, input, target):
        assert len(input) == 2
        emb_q, emb_k = input

        # compute extended contrastive loss only on the embeddings coming from q
        contrastive_loss = super().forward(emb_q, target)

        # TODO enable computing the consistency on all pixels!
        # compute consistency term
        for e_q, e_k, t in zip(emb_q, emb_k, target):
            unlabeled_mask = (t[0] == 0).int()
            if unlabeled_mask.sum() < self.volume_threshold * unlabeled_mask.numel():
                continue
            emb_consistency_loss = self.emb_consistency(e_q, e_k, unlabeled_mask)
            contrastive_loss += self.consistency_term_weight * emb_consistency_loss

        return contrastive_loss


# FIXME clarify what this is!
class SPOCOConsistencyLoss(nn.Module):
    def __init__(self, delta_var, pmaps_threshold, max_anchors=30, norm="fro"):
        super().__init__()
        self.max_anchors = max_anchors
        self.consistency_loss = DiceLoss()
        self.norm = norm
        self.dist_to_mask = GaussianKernel(delta_var=delta_var, pmaps_threshold=pmaps_threshold)
        self.init_kwargs = {"delta_var": delta_var, "pmaps_threshold": pmaps_threshold,
                            "max_anchors": max_anchors, "norm": norm}

    def _inst_pmap(self, emb, anchor):
        # compute distance map
        distance_map = torch.norm(emb - anchor, self.norm, dim=-1)
        # convert distance map to instance pmaps and return
        return self.dist_to_mask(distance_map)

    def emb_consistency(self, emb_q, emb_k):
        inst_q = []
        inst_k = []
        mask = torch.ones(emb_q.shape[1:])
        for i in range(self.max_anchors):
            # get random anchor
            indices = torch.nonzero(mask, as_tuple=True)
            ind = np.random.randint(len(indices[0]))

            q_pmap = self._extract_pmap(emb_q, mask, indices, ind)
            inst_q.append(q_pmap)

            k_pmap = self._extract_pmap(emb_k, mask, indices, ind)
            inst_k.append(k_pmap)

        # stack along channel dim
        inst_q = torch.stack(inst_q)
        inst_k = torch.stack(inst_k)

        loss = self.consistency_loss(inst_q, inst_k)
        return loss

    def _extract_pmap(self, emb, mask, indices, ind):
        if mask.dim() == 2:
            y, x = indices
            anchor = emb[:, y[ind], x[ind]]
            emb = emb.permute(1, 2, 0)
        else:
            z, y, x = indices
            anchor = emb[:, z[ind], y[ind], x[ind]]
            emb = emb.permute(1, 2, 3, 0)

        return self._inst_pmap(emb, anchor)

    def forward(self, emb_q, emb_k):
        contrastive_loss = 0.0
        # compute consistency term
        for e_q, e_k in zip(emb_q, emb_k):
            contrastive_loss += self.emb_consistency(e_q, e_k)
        return contrastive_loss
