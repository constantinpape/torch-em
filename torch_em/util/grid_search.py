import numpy as np
import torch.nn as nn
import xarray

import bioimageio.core

from micro_sam.instance_segmentation import InstanceSegmentationWithDecoder
from micro_sam.evaluation.instance_segmentation import (
    default_grid_search_values_instance_segmentation_with_decoder,
    evaluate_instance_segmentation_grid_search,
    run_instance_segmentation_grid_search,
    _get_range_of_search_values,
)

from ..transform.raw import standardize
from .prediction import predict_with_padding, predict_with_halo
from .segmentation import watershed_from_components


def default_grid_search_values_boundary_based_instance_segmentation(
    threshold1_values=None,
    threshold2_values=None,
    min_size_values=None,
):
    if threshold1_values is None:
        threshold1_values = [0.5, 0.55, 0.6]
    if threshold2_values is None:
        threshold2_values = _get_range_of_search_values(
            [0.5, 0.9], step=0.1
        )
    if min_size_values is None:
        min_size_values = [25, 50, 75, 100, 200]

    return {
        "min_size": min_size_values,
        "threshold1": threshold1_values,
        "threshold2": threshold2_values,
    }


class _InstanceSegmentationBase(InstanceSegmentationWithDecoder):
    def __init__(self, model, preprocess=None, block_shape=None, halo=None):
        self._model = model
        self._preprocess = standardize if preprocess is None else preprocess

        assert (block_shape is None) == (halo is None)
        self._block_shape = block_shape
        self._halo = halo

        self._is_initialized = False

    def _initialize_torch(self, data):
        device = next(iter(self._model.parameters())).device

        if self._block_shape is None:
            if hasattr(self._model, "scale_factors"):
                scale_factors = self._model.init_kwargs["scale_factors"]
                min_divisible = [int(np.prod([sf[i] for sf in scale_factors])) for i in range(3)]
            elif hasattr(self._model, "depth"):
                depth = self._model.depth
                min_divisible = [2**depth, 2**depth]
            else:
                raise RuntimeError
            input_ = self._preprocess(data)
            output = predict_with_padding(self._model, input_, min_divisible, device)
        else:
            output = predict_with_halo(
                data, self._model, [device], self._block_shape, self._halo,
                preprocess=self._preprocess,
            )
        return output

    def _initialize_modelzoo(self, data):
        if self._block_shape is None:
            with bioimageio.core.create_prediction_pipeline(self._model) as pp:
                dims = tuple("bcyx") if data.ndim == 2 else tuple("bczyx")
                input_ = xarray.DataArray(data[None, None], dims=dims)
                output = bioimageio.core.prediction.predict_with_padding(pp, input_, padding=True)[0]
                output = output.squeeze().values
        else:
            raise NotImplementedError
        return output


class BoundaryBasedInstanceSegmentation(_InstanceSegmentationBase):
    def __init__(self, model, preprocess=None, block_shape=None, halo=None):
        super().__init__(
            model=model, preprocess=preprocess, block_shape=block_shape, halo=halo
        )

        self._foreground = None
        self._boundaries = None

    def initialize(self, data):
        if isinstance(self._model, nn.Module):
            output = self._initialize_torch(data)
        else:
            output = self._initialize_modelzoo(data)

        assert output.shape[0] == 2

        self._foreground = output[0]
        self._boundaries = output[1]

        self._is_initialized = True

    def generate(self, min_size=50, threshold1=0.5, threshold2=0.5, output_mode="binary_mask"):
        segmentation = watershed_from_components(
            self._boundaries, self._foreground,
            min_size=min_size, threshold1=threshold1, threshold2=threshold2,
        )
        if output_mode is not None:
            segmentation = self._to_masks(segmentation, output_mode)
        return segmentation


class DistanceBasedInstanceSegmentation(_InstanceSegmentationBase):
    """Over-write micro_sam functionality so that it works for distance based
    segmentation with a U-net.
    """
    def __init__(self, model, preprocess=None, block_shape=None, halo=None):
        super().__init__(
            model=model, preprocess=preprocess, block_shape=block_shape, halo=halo
        )

        self._foreground = None
        self._center_distances = None
        self._boundary_distances = None

    def initialize(self, data):
        if isinstance(self._model, nn.Module):
            output = self._initialize_torch(data)
        else:
            output = self._initialize_modelzoo(data)

        assert output.shape[0] == 3
        self._foreground = output[0]
        self._center_distances = output[1]
        self._boundary_distances = output[2]

        self._is_initialized = True


def instance_segmentation_grid_search(
    segmenter, image_paths, gt_paths, result_dir,
    grid_search_values=None, rois=None,
    image_key=None, gt_key=None,
):
    if grid_search_values is None:
        if isinstance(segmenter, DistanceBasedInstanceSegmentation):
            grid_search_values = default_grid_search_values_instance_segmentation_with_decoder()
        elif isinstance(segmenter, BoundaryBasedInstanceSegmentation):
            grid_search_values = default_grid_search_values_boundary_based_instance_segmentation()
        else:
            raise ValueError(f"Could not derive default grid search values for segmenter of type {type(segmenter)}")

    run_instance_segmentation_grid_search(
        segmenter, grid_search_values, image_paths, gt_paths, result_dir,
        embedding_dir=None, verbose_gs=True,
        image_key=image_key, gt_key=gt_key, rois=rois,
    )
    best_kwargs, best_score = evaluate_instance_segmentation_grid_search(
        result_dir, list(grid_search_values.keys())
    )
    return best_kwargs, best_score
