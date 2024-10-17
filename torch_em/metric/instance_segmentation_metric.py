from functools import partial

import numpy as np
import elf.evaluation as elfval
import elf.segmentation as elfseg
import elf.segmentation.embeddings as elfemb
import torch
import torch.nn as nn
import vigra
from elf.segmentation.watershed import apply_size_filter

print("Hello Anwai, please help")


class BaseInstanceSegmentationMetric(nn.Module):
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
    def __init__(self, threshold=0.5, metric="precision"):
        self.threshold = threshold
        self.metric = metric

    def __call__(self, seg, target):
        score = 1.0 - elfval.matching(seg, target, threshold=self.threshold)[self.metric]
        return score


class VariationOfInformation:
    def __call__(self, seg, target):
        vis, vim = elfval.variation_of_information(seg, target)
        return vis + vim


class AdaptedRandError:
    def __call__(self, seg, target):
        are, _ = elfval.rand_index(seg, target)
        return are


class SymmetricBestDice:
    def __call__(self, seg, target):
        score = 1.0 - elfval.symmetric_best_dice_score(seg, target)
        return score


#
# Prefab Full Metrics
#


class EmbeddingMWSIOUMetric(BaseInstanceSegmentationMetric):
    def __init__(self, delta, offsets, min_seg_size, iou_threshold=0.5, strides=None):
        segmenter = EmbeddingMWS(delta, offsets, with_background=True, min_seg_size=min_seg_size)
        metric = IOUError(iou_threshold)
        super().__init__(segmenter, metric)
        self.init_kwargs = {"delta": delta, "offsets": offsets, "min_seg_size": min_seg_size,
                            "iou_threshold": iou_threshold, "strides": strides}


class EmbeddingMWSSBDMetric(BaseInstanceSegmentationMetric):
    def __init__(self, delta, offsets, min_seg_size, strides=None):
        segmenter = EmbeddingMWS(delta, offsets, with_background=True, min_seg_size=min_seg_size)
        metric = SymmetricBestDice()
        super().__init__(segmenter, metric)
        self.init_kwargs = {"delta": delta, "offsets": offsets, "min_seg_size": min_seg_size, "strides": strides}


class EmbeddingMWSVOIMetric(BaseInstanceSegmentationMetric):
    def __init__(self, delta, offsets, min_seg_size, strides=None):
        segmenter = EmbeddingMWS(delta, offsets, with_background=False, min_seg_size=min_seg_size)
        metric = VariationOfInformation()
        super().__init__(segmenter, metric)
        self.init_kwargs = {"delta": delta, "offsets": offsets, "min_seg_size": min_seg_size, "strides": strides}


class EmbeddingMWSRandMetric(BaseInstanceSegmentationMetric):
    def __init__(self, delta, offsets, min_seg_size, strides=None):
        segmenter = EmbeddingMWS(delta, offsets, with_background=False, min_seg_size=min_seg_size)
        metric = AdaptedRandError()
        super().__init__(segmenter, metric)
        self.init_kwargs = {"delta": delta, "offsets": offsets, "min_seg_size": min_seg_size, "strides": strides}


class HDBScanIOUMetric(BaseInstanceSegmentationMetric):
    def __init__(self, min_size, eps, iou_threshold=0.5):
        segmenter = HDBScan(min_size=min_size, eps=eps, remove_largest=True)
        metric = IOUError(iou_threshold)
        super().__init__(segmenter, metric)
        self.init_kwargs = {"min_size": min_size, "eps": eps, "iou_threshold": iou_threshold}


class HDBScanSBDMetric(BaseInstanceSegmentationMetric):
    def __init__(self, min_size, eps):
        segmenter = HDBScan(min_size=min_size, eps=eps, remove_largest=True)
        metric = SymmetricBestDice()
        super().__init__(segmenter, metric)
        self.init_kwargs = {"min_size": min_size, "eps": eps}


class HDBScanRandMetric(BaseInstanceSegmentationMetric):
    def __init__(self, min_size, eps):
        segmenter = HDBScan(min_size=min_size, eps=eps, remove_largest=True)
        metric = AdaptedRandError()
        super().__init__(segmenter, metric)
        self.init_kwargs = {"min_size": min_size, "eps": eps}


class HDBScanVOIMetric(BaseInstanceSegmentationMetric):
    def __init__(self, min_size, eps):
        segmenter = HDBScan(min_size=min_size, eps=eps, remove_largest=True)
        metric = VariationOfInformation()
        super().__init__(segmenter, metric)
        self.init_kwargs = {"min_size": min_size, "eps": eps}


class MulticutVOIMetric(BaseInstanceSegmentationMetric):
    def __init__(self, min_seg_size, anisotropic=False, dt_threshold=0.25, sigma_seeds=2.0):
        segmenter = Multicut(dt_threshold, anisotropic, sigma_seeds)
        metric = VariationOfInformation()
        super().__init__(segmenter, metric)
        self.init_kwargs = {"anisotropic": anisotropic, "min_seg_size": min_seg_size,
                            "dt_threshold": dt_threshold, "sigma_seeds": sigma_seeds}


class MulticutRandMetric(BaseInstanceSegmentationMetric):
    def __init__(self, min_seg_size, anisotropic=False, dt_threshold=0.25, sigma_seeds=2.0):
        segmenter = Multicut(dt_threshold, anisotropic, sigma_seeds)
        metric = AdaptedRandError()
        super().__init__(segmenter, metric)
        self.init_kwargs = {"anisotropic": anisotropic, "min_seg_size": min_seg_size,
                            "dt_threshold": dt_threshold, "sigma_seeds": sigma_seeds}


class MWSIOUMetric(BaseInstanceSegmentationMetric):
    def __init__(self, offsets, min_seg_size, iou_threshold=0.5, strides=None):
        segmenter = MWS(offsets, with_background=True, min_seg_size=min_seg_size, strides=strides)
        metric = IOUError(iou_threshold)
        super().__init__(segmenter, metric)
        self.init_kwargs = {"offsets": offsets, "min_seg_size": min_seg_size,
                            "iou_threshold": iou_threshold, "strides": strides}


class MWSSBDMetric(BaseInstanceSegmentationMetric):
    def __init__(self, offsets, min_seg_size, strides=None):
        segmenter = MWS(offsets, with_background=True, min_seg_size=min_seg_size, strides=strides)
        metric = SymmetricBestDice()
        super().__init__(segmenter, metric)
        self.init_kwargs = {"offsets": offsets, "min_seg_size": min_seg_size, "strides": strides}


class MWSVOIMetric(BaseInstanceSegmentationMetric):
    def __init__(self, offsets, min_seg_size, strides=None):
        segmenter = MWS(offsets, with_background=False, min_seg_size=min_seg_size, strides=strides)
        metric = VariationOfInformation()
        super().__init__(segmenter, metric)
        self.init_kwargs = {"offsets": offsets, "min_seg_size": min_seg_size, "strides": strides}


class MWSRandMetric(BaseInstanceSegmentationMetric):
    def __init__(self, offsets, min_seg_size, strides=None):
        segmenter = MWS(offsets, with_background=False, min_seg_size=min_seg_size, strides=strides)
        metric = AdaptedRandError()
        super().__init__(segmenter, metric)
        self.init_kwargs = {"offsets": offsets, "min_seg_size": min_seg_size, "strides": strides}
