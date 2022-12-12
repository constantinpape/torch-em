# TODO refactor this all into elf
# (or rather refactor the more general versions from the embedding clustering repo)
import hdbscan
import numpy as np
import vigra

import elf.segmentation.features as efeats
import elf.segmentation.mutex_watershed as emws


def segment_hdbscan(emb, semantic_mask=None, remove_largest=True):
    output_shape = emb.shape[1:]
    # reshape (E, D, H, W) -> (E, D * H * W) and transpose -> (D * H * W, E)
    flattened_embeddings = emb.reshape(emb.shape[0], -1).transpose()

    result = np.zeros(flattened_embeddings.shape[0])

    if semantic_mask is not None:
        flattened_mask = semantic_mask.reshape(-1)
        assert flattened_mask.shape[0] == flattened_embeddings.shape[0]
    else:
        flattened_mask = np.ones(flattened_embeddings.shape[0])

    if flattened_mask.sum() == 0:
        # return zeros for empty masks
        return result.reshape(output_shape)

    # cluster only within the foreground mask
    clustering_alg = hdbscan.HDBSCAN(min_cluster_size=140, cluster_selection_epsilon=0.5)
    clusters = clustering_alg.fit_predict(flattened_embeddings[flattened_mask == 1])
    # always increase the labels by 1 cause clustering results start from 0 and we may loose one object
    result[flattened_mask == 1] = clusters + 1

    if remove_largest:
        # set largest object to 0-label
        ids, counts = np.unique(result, return_counts=True)
        result[ids[np.argmax(counts)] == result] = 0

    return result.reshape(output_shape).astype("uint64")


def _get_lr_offsets(offsets):
    lr_offsets = [
        off for off in offsets if np.sum(np.abs(off)) > 1
    ]
    return lr_offsets


def _embeddings_to_problem(embed, distance_type, offsets, strides):
    im_shape = embed.shape[1:]
    g = efeats.compute_grid_graph(im_shape)
    _, weights = efeats.compute_grid_graph_image_features(g, embed, distance_type)
    lr_offsets = _get_lr_offsets(offsets)
    lr_edges, lr_weights = efeats.compute_grid_graph_image_features(
        g, embed, distance_type, offsets=lr_offsets,
        strides=strides, randomize_strides=True
    )
    return g, weights, lr_edges, lr_weights


def segment_embeddings_mws(emb, offsets, distance_type, strides=None, min_size=100):
    strides = [3, 3] if strides is None else strides
    g, costs, mutex_uvs, mutex_costs = _embeddings_to_problem(emb, distance_type, offsets=offsets, strides=strides)
    uvs = g.uvIds()
    seg = emws.mutex_watershed_clustering(uvs, mutex_uvs, costs, mutex_costs).reshape(emb.shape[1:])
    if min_size > 0:
        seg_ids, seg_counts = np.unique(seg, return_counts=True)
        remove_ids = seg_ids[seg_counts < min_size]
        seg[np.isin(seg, remove_ids)] = 0
        vigra.analysis.relabelConsecutive(seg)
    return seg
