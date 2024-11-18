"""The CoNSeP dataset contains annotations for nucleus segmentation in
H&E stained histopathology images for multi-tissue regions.

NOTE: The source of this dataset is an open-source version hosted on Kaggle:
- https://www.kaggle.com/datasets/rftexas/tiled-consep-224x224px

This dataset is from the publication https://doi.org/10.1016/j.media.2019.101563.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, List, Literal

import numpy as np
from scipy.io import loadmat
import imageio.v3 as imageio
from skimage.measure import label as connected_components

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


def _stitch_segmentation(segmentation, tile_shape):
    import nifty.tools as nt
    from nifty.ground_truth import overlap

    from elf.segmentation.features import compute_rag, project_node_labels_to_pixels
    from elf.segmentation.multicut import compute_edge_costs, multicut_decomposition

    shape = segmentation.shape
    ndim = len(shape)
    blocking = nt.blocking([0] * ndim, shape, tile_shape)
    n_blocks = blocking.numberOfBlocks
    halo = (1, 1)

    block_segs = []
    for block_id in tqdm(range(n_blocks), desc="Get tiles"):
        block = blocking.getBlockWithHalo(block_id, list(halo))
        bb = tuple(slice(beg, end) for beg, end in zip(block.outerBlock.begin, block.outerBlock.end))
        block_seg = segmentation[bb]
        block_segs.append(block_seg)

    seg_ids = np.unique(segmentation)
    rag = compute_rag(segmentation)
    edge_disaffinities = np.full(rag.numberOfEdges, 0.9, dtype="float32")

    for block_id in tqdm(range(n_blocks), desc="Stitch tiles with overlap"):
        for axis in range(ndim):
            ngb_id = blocking.getNeighborId(block_id, axis, lower=True)
            if ngb_id == -1:
                continue

            this_block = blocking.getBlockWithHalo(block_id, list(halo))
            ngb_block = blocking.getBlockWithHalo(ngb_id, list(halo))

            this_seg, ngb_seg = block_segs[block_id], block_segs[ngb_id]

            face = tuple(
                slice(beg_out, end_out) if d != axis else slice(beg_out, beg_in + halo[d])
                for d, (beg_out, end_out, beg_in) in enumerate(
                    zip(this_block.outerBlock.begin, this_block.outerBlock.end, this_block.innerBlock.begin)
                )
            )

            this_face_bb = tuple(
                slice(fa.start - offset, fa.stop - offset) for fa, offset in zip(face, this_block.outerBlock.begin)
            )
            ngb_face_bb = tuple(
                slice(fa.start - offset, fa.stop - offset) for fa, offset in zip(face, ngb_block.outerBlock.begin)
            )
            this_face = this_seg[this_face_bb]
            ngb_face = ngb_seg[ngb_face_bb]
            assert this_face.shape == ngb_face.shape, (this_face.shape, ngb_face.shape)

            # COMMENT: I visualized the faces, seems like they look as expected.

            # NOTE: I have a feeling that the overlap method is either not working for here or I am missing something.
            overlap_comp = overlap(this_face, ngb_face)
            this_ids = np.unique(this_face)
            overlaps = {this_id: overlap_comp.overlapArraysNormalized(this_id, sorted=False) for this_id in this_ids}
            overlap_ids = {this_id: ovlps[0] for this_id, ovlps in overlaps.items()}
            overlap_values = {this_id: ovlps[1] for this_id, ovlps in overlaps.items()}
            overlap_uv_ids = np.array([
                [this_id, ovlp_id] for this_id, ovlp_ids in overlap_ids.items() for ovlp_id in ovlp_ids
            ])
            overlap_values = np.array([ovlp for ovlps in overlap_values.values() for ovlp in ovlps], dtype="float32")
            assert len(overlap_uv_ids) == len(overlap_values)

            valid_uv_ids = np.isin(overlap_uv_ids, seg_ids).all(axis=1)
            if valid_uv_ids.sum() == 0:
                continue
            overlap_uv_ids, overlap_values = overlap_uv_ids[valid_uv_ids], overlap_values[valid_uv_ids]
            assert len(overlap_uv_ids) == len(overlap_values)

            edge_ids = rag.findEdges(overlap_uv_ids)
            # NOTE: All edges I receive seem to be '-1' (something is wrong with computing 'overlap_uv_ids')
            valid_edges = edge_ids != -1
            if valid_edges.sum() == 0:
                continue
            edge_ids, overlap_values = edge_ids[valid_edges], overlap_values[valid_edges]
            assert len(edge_ids) == len(overlap_values)

            edge_disaffinities[edge_ids] = (1.0 - overlap_values)

    costs = compute_edge_costs(edge_disaffinities, beta=0.5)

    node_labels = multicut_decomposition(rag, costs)
    seg_stitched = project_node_labels_to_pixels(rag, node_labels)

    import napari
    v = napari.Viewer()
    v.add_labels(seg_stitched)
    napari.run()


def _preprocess_data(data_dir, split):
    import h5py

    preprocessed_dir = os.path.join(data_dir, "preprocessed", split)
    os.makedirs(preprocessed_dir, exist_ok=True)
    for i in tqdm(range(1, 28 if split == "train" else 15), desc="Preprocessing inputs"):
        raw_paths = natsorted(glob(os.path.join(data_dir, "tiles", f"{split}_{i}_*.png")))
        label_paths = [p.replace("tiles", "labels").replace(".png", ".mat") for p in raw_paths]

        raw_tiles, label_tiles, tile_shapes = [], [], []
        for rpath, lpath in zip(raw_paths, label_paths):
            tile_shapes.append(tuple(int(t) for t in Path(rpath).stem.split("_")[2:]))
            raw_tiles.append(imageio.imread(rpath))
            label_tiles.append(loadmat(lpath)["instance_map"])

        h = max(shape[1] for shape in tile_shapes)
        w = max(shape[3] for shape in tile_shapes)

        labels = np.zeros((h, w))

        raw = np.zeros((h, w, 3))
        for tile, shape in zip(raw_tiles, tile_shapes):
            y1, y2, x1, x2 = shape
            raw[y1: y2, x1: x2] = tile

        labels = np.zeros((h, w))
        for tile, shape in zip(label_tiles, tile_shapes):
            y1, y2, x1, x2 = shape
            labels[y1: y2, x1: x2] = tile

        labels = connected_components(labels).astype("uint32")
        labels = _stitch_segmentation(labels, tile_shape=(224, 224))

        volume_path = os.path.join(preprocessed_dir, f"{i}.h5")
        with h5py.File(volume_path, "w") as f:
            f.create_dataset("raw", data=raw.transpose(2, 0, 1), compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")


def get_consep_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the CoNSeP dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded and preprocessed.
    """
    data_dir = os.path.join(path, "data", "consep")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    util.download_source_kaggle(path=path, dataset_name="rftexas/tiled-consep-224x224px", download=download)
    util.unzip(zip_path=os.path.join(
        path, "tiled-consep-224x224px.zip"), dst=os.path.join(path, "data"), remove=False
    )

    return data_dir


def get_consep_paths(
    path: Union[os.PathLike, str], split: Literal['train', 'test'], download: bool = False
) -> List[str]:
    """Get paths to the CoNSeP data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the input data.
    """
    data_dir = get_consep_data(path, download)

    _preprocess_data(data_dir, "train")
    _preprocess_data(data_dir, "test")

    if split not in ['train', 'test']:
        raise ValueError(f"'{split}' is not a valid split.")

    volume_paths = natsorted(glob(os.path.join(data_dir, "preprocessed", split, "*.h5")))
    return volume_paths


def get_consep_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal['train', 'test'],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CoNSeP dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_consep_paths(path, split, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key="labels",
        is_seg_dataset=True,
        patch_shape=patch_shape,
        with_channels=True,
        ndim=2,
        **kwargs
    )


def get_consep_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal['train', 'test'],
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CoNSeP dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_consep_dataset(path, patch_shape, split, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
