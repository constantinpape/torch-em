from functools import partial

import numpy as np
import elf.segmentation.embeddings as elfemb
import torch
import torch.nn as nn
import vigra

from elf.evaluation import matching


class BaseInstanceSegmentationMetric(nn.Module):
    def __init__(self, segmenter, metric, to_numpy=True):
        super().__init__()
        self.segmenter = segmenter
        self.metric = metric
        self.to_numpy = to_numpy

    def forward(self, input_, target):
        if self.to_numpy:
            input_ = input_.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
        assert input_.ndim == target.ndim
        assert len(input_) == len(target)
        scores = []
        # compute the metric per batch
        for pred, trgt in zip(input_, target):
            seg = self.segmenter(pred)
            # by convention we assume that the segmentation channel is always in the last channel of trgt
            scores.append(self.metric(seg, trgt[-1]))
        return torch.tensor(scores).mean()

    @staticmethod
    def filter_sizes(seg, min_seg_size, hmap=None):
        seg_ids, counts = np.unique(seg, return_counts=True)
        if hmap is None:
            bg_ids = seg_ids[counts < min_seg_size]
            seg[np.isin(seg, bg_ids)] = 0
        else:
            raise NotImplementedError  # TODO
        return seg


# TODO normal mws for affinites, multicut for boundaries, hdbscan for embeddings
#
# Segmenters
#

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

    def merge_embeddings(self, seg, embeddings):
        seg += 1
        seg_ids, counts = np.unique(seg, return_counts=True)
        bg_seg = seg_ids[np.argmax(counts)]
        mean_embeddings = np.concatenate([
            vigra.analysis.extractRegionFeatures(emb, seg.astype("uint32"), features=["mean"])["mean"][None]
            for emb in embeddings
        ], axis=0)
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
            seg = self.filter_sizes(seg, self.min_seg_size)
        return seg


#
# Metrics
#

class IOUError:
    def __init__(self, threshold=0.5, metric="precision"):
        self.threshold = threshold
        self.metric = metric

    def __call__(self, seg, target):
        return 1.0 - matching(seg, target, threshold=self.threshold)[self.metric]


# TODO
class VariationOfInformation:
    pass


# TODO
class AdaptedRandError:
    pass


#
# Ready made metrics
#


class EmbeddingMWSIOUMetric(BaseInstanceSegmentationMetric):
    def __init__(self, delta, offsets, min_seg_size, iou_threshold, strides=None):
        segmenter = EmbeddingMWS(delta, offsets, with_background=True, min_seg_size=min_seg_size)
        metric = IOUError(iou_threshold)
        super().__init__(segmenter, metric)
        self.init_kwargs = {"delta": delta, "offsets": offsets, "min_seg_size": min_seg_size,
                            "iou_threshold": iou_threshold, "strides": strides}
