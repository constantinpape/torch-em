from functools import partial
from typing import List, Optional

import numpy as np
import elf.evaluation as elfval
import elf.segmentation as elfseg
import elf.segmentation.embeddings as elfemb
import torch
import torch.nn as nn
import vigra
from elf.segmentation.watershed import apply_size_filter


class BaseInstanceSegmentationMetric(nn.Module):
    """@private
    """
    def __init__(self, segmenter, metric, to_numpy=True):
        super().__init__()
        self.segmenter = segmenter
        self.metric = metric
        self.to_numpy = to_numpy

    def forward(self, input_, target):
        if self.to_numpy:
            input_ = input_.detach().cpu().numpy().astype("float32")
            target = target.detach().cpu().numpy()
        assert input_.ndim == target.ndim
        assert len(input_) == len(target)
        scores = []
        # compute the metric per batch
        for pred, trgt in zip(input_, target):
            seg = self.segmenter(pred)
            # by convention we assume that the segmentation channel is always in the last channel of trgt
            scores.append(self.metric(seg, trgt[-1].astype("uint32")))
        return torch.tensor(scores).mean()


#
# Segmenters
#

def filter_sizes(seg, min_seg_size, hmap=None):
    """@private
    """
    seg_ids, counts = np.unique(seg, return_counts=True)
    if hmap is None:
        bg_ids = seg_ids[counts < min_seg_size]
        seg[np.isin(seg, bg_ids)] = 0
    else:
        ndim = seg.ndim
        hmap_ = hmap if hmap.ndim == ndim else np.max(hmap, axis=0)
        seg, _ = apply_size_filter(seg, hmap_, min_seg_size)
    return seg


class MWS:
    """@private
    """
    def __init__(self, offsets, with_background, min_seg_size, strides=None):
        self.offsets = offsets
        self.with_background = with_background
        self.min_seg_size = min_seg_size
        if strides is None:
            strides = [4] * len(offsets[0])
        assert len(strides) == len(offsets[0])
        self.strides = strides

    def __call__(self, affinities):
        if self.with_background:
            assert len(affinities) == len(self.offsets) + 1
            mask, affinities = affinities[0], affinities[1:]
        else:
            assert len(affinities) == len(self.offsets)
            mask = None
        seg = elfseg.mutex_watershed.mutex_watershed(affinities, self.offsets, self.strides,
                                                     randomize_strides=True, mask=mask).astype("uint32")
        if self.min_seg_size > 0:
            seg = filter_sizes(seg, self.min_seg_size,
                               hmap=None if self.with_background else affinities)
        return seg


class EmbeddingMWS:
    """@private
    """
    def __init__(self, delta, offsets, with_background, min_seg_size, strides=None):
        self.delta = delta
        self.offsets = offsets
        self.with_background = with_background
        self.min_seg_size = min_seg_size
        if strides is None:
            strides = [4] * len(offsets[0])
        assert len(strides) == len(offsets[0])
        self.strides = strides

    def merge_background(self, seg, embeddings):
        seg += 1
        seg_ids, counts = np.unique(seg, return_counts=True)
        bg_seg = seg_ids[np.argmax(counts)]
        mean_embeddings = []
        for emb in embeddings:
            mean_embeddings.append(vigra.analysis.extractRegionFeatures(emb, seg, features=["mean"])["mean"][None])
        mean_embeddings = np.concatenate(mean_embeddings, axis=0)
        bg_embed = mean_embeddings[:, bg_seg][:, None]
        bg_probs = elfemb._embeddings_to_probabilities(mean_embeddings, bg_embed, self.delta, 0)
        bg_ids = np.where(bg_probs > 0.5)
        seg[np.isin(seg, bg_ids)] = 0
        vigra.analysis.relabelConsecutive(seg, out=seg)
        return seg

    def __call__(self, embeddings):
        weight = partial(elfemb.discriminative_loss_weight, delta=self.delta)
        seg = elfemb.segment_embeddings_mws(
            embeddings, "l2", self.offsets, strides=self.strides, weight_function=weight
        ).astype("uint32")
        if self.with_background:
            seg = self.merge_background(seg, embeddings)
        if self.min_seg_size > 0:
            seg = filter_sizes(seg, self.min_seg_size)
        return seg


class Multicut:
    """@private
    """
    def __init__(self, min_seg_size, anisotropic=False, dt_threshold=0.25, sigma_seeds=2.0, solver="decomposition"):
        self.min_seg_size = min_seg_size
        self.anisotropic = anisotropic
        self.dt_threshold = dt_threshold
        self.sigma_seeds = sigma_seeds
        self.solver = solver

    def __call__(self, boundaries):
        if boundaries.shape[0] == 1:
            boundaries = boundaries[0]
        assert boundaries.ndim in (2, 3), f"{boundaries.ndim}"
        if self.anisotropic and boundaries.ndim == 3:
            ws, max_id = elfseg.stacked_watershed(boundaries, threshold=self.dt_threshold,
                                                  sigma_seed=self.sigma_seeds,
                                                  sigma_weights=self.sigma_seeds,
                                                  n_threads=1)
        else:
            ws, max_id = elfseg.distance_transform_watershed(boundaries, threshold=self.dt_threshold,
                                                             sigma_seeds=self.sigma_seeds,
                                                             sigma_weights=self.sigma_seeds)
        rag = elfseg.compute_rag(ws, max_id + 1, n_threads=1)
        feats = elfseg.compute_boundary_mean_and_length(rag, boundaries, n_threads=1)[:, 0]
        costs = elfseg.compute_edge_costs(feats)
        solver = elfseg.get_multicut_solver(self.solver)
        node_labels = solver(rag, costs, n_threads=1)
        seg = elfseg.project_node_labels_to_pixels(rag, node_labels, n_threads=1).astype("uint32")
        if self.min_seg_size > 0:
            seg = filter_sizes(seg, self.min_seg_size, hmap=boundaries)
        return seg


class HDBScan:
    """@private
    """
    def __init__(self, min_size, eps, remove_largest):
        self.min_size = min_size
        self.eps = eps
        self.remove_largest = remove_largest

    def __call__(self, embeddings):
        return elfemb.segment_hdbscan(embeddings, self.min_size, self.eps, self.remove_largest)


#
# Metrics
#

class IOUError:
    """@private
    """
    def __init__(self, threshold=0.5, metric="precision"):
        self.threshold = threshold
        self.metric = metric

    def __call__(self, seg, target):
        score = 1.0 - elfval.matching(seg, target, threshold=self.threshold)[self.metric]
        return score


class VariationOfInformation:
    """@private
    """
    def __call__(self, seg, target):
        vis, vim = elfval.variation_of_information(seg, target)
        return vis + vim


class AdaptedRandError:
    """@private
    """
    def __call__(self, seg, target):
        are, _ = elfval.rand_index(seg, target)
        return are


class SymmetricBestDice:
    """@private
    """
    def __call__(self, seg, target):
        score = 1.0 - elfval.symmetric_best_dice_score(seg, target)
        return score


#
# Prefab Full Metrics
#


class EmbeddingMWSIOUMetric(BaseInstanceSegmentationMetric):
    """Intersection over union metric based on mutex watershed computed from embedding-derived affinites.

    This class can be used as validation metric when training a network for instance segmentation.

    Args:
        delta: The hinge distance of the contrastive loss for training the embeddings.
        offsets: The offsets for deriving the affinities from the embeddings.
        min_seg_size: Size for filtering the segmentation objects.
        iou_threshold: Threshold for the intersection over union metric.
        strides: The strides for the mutex watershed.
    """
    def __init__(
        self,
        delta: float,
        offsets: List[List[int]],
        min_seg_size: int,
        iou_threshold: float = 0.5,
        strides: Optional[List[int]] = None,
    ):
        segmenter = EmbeddingMWS(delta, offsets, with_background=True, min_seg_size=min_seg_size)
        metric = IOUError(iou_threshold)
        super().__init__(segmenter, metric)
        self.init_kwargs = {"delta": delta, "offsets": offsets, "min_seg_size": min_seg_size,
                            "iou_threshold": iou_threshold, "strides": strides}


class EmbeddingMWSSBDMetric(BaseInstanceSegmentationMetric):
    """Symmetric best dice metric based on mutex watershed computed from embedding-derived affinites.

    This class can be used as validation metric when training a network for instance segmentation.

    Args:
        delta: The hinge distance of the contrastive loss for training the embeddings.
        offsets: The offsets for deriving the affinities from the embeddings.
        min_seg_size: Size for filtering the segmentation objects.
        strides: The strides for the mutex watershed.
    """
    def __init__(self, delta: float, offsets: List[List[int]], min_seg_size: int, strides: Optional[List[int]] = None):
        segmenter = EmbeddingMWS(delta, offsets, with_background=True, min_seg_size=min_seg_size)
        metric = SymmetricBestDice()
        super().__init__(segmenter, metric)
        self.init_kwargs = {"delta": delta, "offsets": offsets, "min_seg_size": min_seg_size, "strides": strides}


class EmbeddingMWSVOIMetric(BaseInstanceSegmentationMetric):
    """Variation of inofrmation metric based on mutex watershed computed from embedding-derived affinites.

    This class can be used as validation metric when training a network for instance segmentation.

    Args:
        delta: The hinge distance of the contrastive loss for training the embeddings.
        offsets: The offsets for deriving the affinities from the embeddings.
        min_seg_size: Size for filtering the segmentation objects.
        strides: The strides for the mutex watershed.
    """
    def __init__(self, delta: float, offsets: List[List[int]], min_seg_size: int, strides: Optional[List[int]] = None):
        segmenter = EmbeddingMWS(delta, offsets, with_background=False, min_seg_size=min_seg_size)
        metric = VariationOfInformation()
        super().__init__(segmenter, metric)
        self.init_kwargs = {"delta": delta, "offsets": offsets, "min_seg_size": min_seg_size, "strides": strides}


class EmbeddingMWSRandMetric(BaseInstanceSegmentationMetric):
    """Rand index metric based on mutex watershed computed from embedding-derived affinites.

    This class can be used as validation metric when training a network for instance segmentation.

    Args:
        delta: The hinge distance of the contrastive loss for training the embeddings.
        offsets: The offsets for deriving the affinities from the embeddings.
        min_seg_size: Size for filtering the segmentation objects.
        strides: The strides for the mutex watershed.
    """
    def __init__(self, delta: float, offsets: List[List[int]], min_seg_size: int, strides: Optional[List[int] ] = None):
        segmenter = EmbeddingMWS(delta, offsets, with_background=False, min_seg_size=min_seg_size)
        metric = AdaptedRandError()
        super().__init__(segmenter, metric)
        self.init_kwargs = {"delta": delta, "offsets": offsets, "min_seg_size": min_seg_size, "strides": strides}


class HDBScanIOUMetric(BaseInstanceSegmentationMetric):
    """Intersection over union metric based on HDBScan computed from embeddings.

    This class can be used as validation metric when training a network for instance segmentation.

    Args:
        min_size: The minimal segment size.
        eps: The epsilon value for HDBScan.
        iou_threshold: The threshold for the intersection over union value.
    """
    def __init__(self, min_size: int, eps: float, iou_threshold: float = 0.5):
        segmenter = HDBScan(min_size=min_size, eps=eps, remove_largest=True)
        metric = IOUError(iou_threshold)
        super().__init__(segmenter, metric)
        self.init_kwargs = {"min_size": min_size, "eps": eps, "iou_threshold": iou_threshold}


class HDBScanSBDMetric(BaseInstanceSegmentationMetric):
    """Symmetric best dice metric based on HDBScan computed from embeddings.

    This class can be used as validation metric when training a network for instance segmentation.

    Args:
        min_size: The minimal segment size.
        eps: The epsilon value for HDBScan.
    """
    def __init__(self, min_size: int, eps: float):
        segmenter = HDBScan(min_size=min_size, eps=eps, remove_largest=True)
        metric = SymmetricBestDice()
        super().__init__(segmenter, metric)
        self.init_kwargs = {"min_size": min_size, "eps": eps}


class HDBScanRandMetric(BaseInstanceSegmentationMetric):
    """Rand index metric based on HDBScan computed from embeddings.

    This class can be used as validation metric when training a network for instance segmentation.

    Args:
        min_size: The minimal segment size.
        eps: The epsilon value for HDBScan.
    """
    def __init__(self, min_size: int, eps: float):
        segmenter = HDBScan(min_size=min_size, eps=eps, remove_largest=True)
        metric = AdaptedRandError()
        super().__init__(segmenter, metric)
        self.init_kwargs = {"min_size": min_size, "eps": eps}


class HDBScanVOIMetric(BaseInstanceSegmentationMetric):
    """Variation of information metric based on HDBScan computed from embeddings.

    This class can be used as validation metric when training a network for instance segmentation.

    Args:
        min_size: The minimal segment size.
        eps: The epsilon value for HDBScan.
    """
    def __init__(self, min_size: int, eps: float):
        segmenter = HDBScan(min_size=min_size, eps=eps, remove_largest=True)
        metric = VariationOfInformation()
        super().__init__(segmenter, metric)
        self.init_kwargs = {"min_size": min_size, "eps": eps}


class MulticutVOIMetric(BaseInstanceSegmentationMetric):
    """Variation of information metric based on a multicut computed from boundary predictions.

    This class can be used as validation metric when training a network for instance segmentation.

    Args:
        min_seg_size: The minimal segment size.
        anisotropic: Whether to compute the watersheds in 2d for volumetric data.
        dt_threshold: The threshold to apply to the boundary predictions before computing the distance transform.
        sigma_seeds: The sigma value for smoothing the distance transform before computing seeds.
    """
    def __init__(self, min_seg_size: int, anisotropic: bool = False, dt_threshold: float = 0.25, sigma_seeds: float = 2.0):
        segmenter = Multicut(dt_threshold, anisotropic, sigma_seeds)
        metric = VariationOfInformation()
        super().__init__(segmenter, metric)
        self.init_kwargs = {"anisotropic": anisotropic, "min_seg_size": min_seg_size,
                            "dt_threshold": dt_threshold, "sigma_seeds": sigma_seeds}


class MulticutRandMetric(BaseInstanceSegmentationMetric):
    """Rand index metric based on a multicut computed from boundary predictions.

    This class can be used as validation metric when training a network for instance segmentation.

    Args:
        min_seg_size: The minimal segment size.
        anisotropic: Whether to compute the watersheds in 2d for volumetric data.
        dt_threshold: The threshold to apply to the boundary predictions before computing the distance transform.
        sigma_seeds: The sigma value for smoothing the distance transform before computing seeds.
    """
    def __init__(self, min_seg_size: int, anisotropic: bool = False, dt_threshold: float = 0.25, sigma_seeds: float = 2.0):
        segmenter = Multicut(dt_threshold, anisotropic, sigma_seeds)
        metric = AdaptedRandError()
        super().__init__(segmenter, metric)
        self.init_kwargs = {"anisotropic": anisotropic, "min_seg_size": min_seg_size,
                            "dt_threshold": dt_threshold, "sigma_seeds": sigma_seeds}


class MWSIOUMetric(BaseInstanceSegmentationMetric):
    """Intersection over union metric based on a mutex watershed computed from affinity predictions.

    This class can be used as validation metric when training a network for instance segmentation.

    Args:
        offsets: The offsets corresponding to the affinity channels.
        min_seg_size: The minimal segment size.
        iou_threshold: The threshold for the intersection over union value.
        strides: The strides for the mutex watershed.
    """
    def __init__(self, offsets: List[List[int]], min_seg_size: int, iou_threshold: float = 0.5, strides: Optional[List[int]] = None):
        segmenter = MWS(offsets, with_background=True, min_seg_size=min_seg_size, strides=strides)
        metric = IOUError(iou_threshold)
        super().__init__(segmenter, metric)
        self.init_kwargs = {"offsets": offsets, "min_seg_size": min_seg_size,
                            "iou_threshold": iou_threshold, "strides": strides}


class MWSSBDMetric(BaseInstanceSegmentationMetric):
    """Symmetric best dice score metric based on a mutex watershed computed from affinity predictions.

    This class can be used as validation metric when training a network for instance segmentation.

    Args:
        offsets: The offsets corresponding to the affinity channels.
        min_seg_size: The minimal segment size.
        strides: The strides for the mutex watershed.
    """
    def __init__(self, offsets: List[List[int]], min_seg_size: int, strides: Optional[List[int]] = None):
        segmenter = MWS(offsets, with_background=True, min_seg_size=min_seg_size, strides=strides)
        metric = SymmetricBestDice()
        super().__init__(segmenter, metric)
        self.init_kwargs = {"offsets": offsets, "min_seg_size": min_seg_size, "strides": strides}


class MWSVOIMetric(BaseInstanceSegmentationMetric):
    """Variation of information metric based on a mutex watershed computed from affinity predictions.

    This class can be used as validation metric when training a network for instance segmentation.

    Args:
        offsets: The offsets corresponding to the affinity channels.
        min_seg_size: The minimal segment size.
        strides: The strides for the mutex watershed.
    """
    def __init__(self, offsets: List[List[int]], min_seg_size: int, strides: Optional[List[int]] = None):
        segmenter = MWS(offsets, with_background=False, min_seg_size=min_seg_size, strides=strides)
        metric = VariationOfInformation()
        super().__init__(segmenter, metric)
        self.init_kwargs = {"offsets": offsets, "min_seg_size": min_seg_size, "strides": strides}


class MWSRandMetric(BaseInstanceSegmentationMetric):
    """Rand index metric based on a mutex watershed computed from affinity predictions.

    This class can be used as validation metric when training a network for instance segmentation.

    Args:
        offsets: The offsets corresponding to the affinity channels.
        min_seg_size: The minimal segment size.
        strides: The strides for the mutex watershed.
    """
    def __init__(self, offsets: List[Listt[int]], min_seg_size: int, strides: Optional[List[int]] = None):
        segmenter = MWS(offsets, with_background=False, min_seg_size=min_seg_size, strides=strides)
        metric = AdaptedRandError()
        super().__init__(segmenter, metric)
        self.init_kwargs = {"offsets": offsets, "min_seg_size": min_seg_size, "strides": strides}
