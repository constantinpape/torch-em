from typing import Callable, Dict, List, Optional, Tuple

import bioimageio.core
import numpy as np
import torch.nn as nn
import xarray

try:
    from micro_sam.instance_segmentation import InstanceSegmentationWithDecoder
    from micro_sam.evaluation.instance_segmentation import (
        default_grid_search_values_instance_segmentation_with_decoder,
        evaluate_instance_segmentation_grid_search,
        run_instance_segmentation_grid_search,
        _get_range_of_search_values,
    )

    HAVE_MICRO_SAM = True
except ImportError:
    class InstanceSegmentationWithDecoder:
        def __init__(self, *args, **kwargs):
            pass

    HAVE_MICRO_SAM = False

from ..transform.raw import standardize
from .prediction import predict_with_padding, predict_with_halo
from .segmentation import watershed_from_components


def default_grid_search_values_boundary_based_instance_segmentation(
    threshold1_values=None, threshold2_values=None, min_size_values=None,
):
    """@private
    """
    if threshold1_values is None:
        threshold1_values = [0.5, 0.55, 0.6]
    if threshold2_values is None:
        threshold2_values = _get_range_of_search_values(
            [0.5, 0.9], step=0.1
        )
    if min_size_values is None:
        min_size_values = [25, 50, 75, 100, 200]

    return {"min_size": min_size_values, "threshold1": threshold1_values, "threshold2": threshold2_values}


class _InstanceSegmentationBase(InstanceSegmentationWithDecoder):
    """Over-write micro_sam functionality so that it works for distance based segmentation with a U-net.
    """
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
    """Wrapper for boundary based instance segmentation.

    Instances of this class can be passed to `instance_segmentation_grid_search`.

    Args:
        model: The model to evaluate. It must predict two channels:
            The first channel fpr foreground probabilities and the second for boundary probabilities.
        preprocess: Optional preprocessing function to apply to the model inputs.
        block_shape: Optional block shape for tiled prediction. If None, the inputs will be predicted en bloc.
        halo: Halo for tiled prediction.
    """
    def __init__(
        self,
        model: nn.Module,
        preprocess: Optional[Callable] = None,
        block_shape: Tuple[int, ...] = None,
        halo: Tuple[int, ...] = None,
    ):
        super().__init__(model=model, preprocess=preprocess, block_shape=block_shape, halo=halo)
        self._foreground = None
        self._boundaries = None

    def initialize(self, data):
        """@private
        """
        if isinstance(self._model, nn.Module):
            output = self._initialize_torch(data)
        else:
            output = self._initialize_modelzoo(data)
        assert output.shape[0] == 2

        self._foreground = output[0]
        self._boundaries = output[1]
        self._is_initialized = True

    def generate(self, min_size=50, threshold1=0.5, threshold2=0.5, output_mode="binary_mask"):
        """@private
        """
        segmentation = watershed_from_components(
            self._boundaries, self._foreground,
            min_size=min_size, threshold1=threshold1, threshold2=threshold2,
        )
        if output_mode is not None:
            segmentation = self._to_masks(segmentation, output_mode)
        return segmentation


class DistanceBasedInstanceSegmentation(_InstanceSegmentationBase):
    """Wrapper for distance based instance segmentation.

    Instances of this class can be passed to `instance_segmentation_grid_search`.

    Args:
        model: The model to evaluate. It must predict three channels:
            The first channel fpr foreground probabilities, the second for center distances
            and the third for boundary distances.
        preprocess: Optional preprocessing function to apply to the model inputs.
        block_shape: Optional block shape for tiled prediction. If None, the inputs will be predicted en bloc.
        halo: Halo for tiled prediction.
    """
    def __init__(
        self,
        model: nn.Module,
        preprocess: Optional[Callable] = None,
        block_shape: Tuple[int, ...] = None,
        halo: Tuple[int, ...] = None,
    ):
        super().__init__(model=model, preprocess=preprocess, block_shape=block_shape, halo=halo)

        self._foreground = None
        self._center_distances = None
        self._boundary_distances = None

    def initialize(self, data):
        """@private
        """
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
    segmenter,
    image_paths: List[str],
    gt_paths: List[str],
    result_dir: str,
    grid_search_values: Optional[Dict] = None,
    rois: Optional[List[Tuple[slice, ...]]] = None,
    image_key: Optional[str] = None,
    gt_key: Optional[str] = None,
) -> Tuple[Dict, float]:
    """Run grid search for instance segmentation.

    Args:
        segmenter: The segmentation wrapper. Needs to provide a 'initialize' and 'generate' function.
            The class `DistanceBasedInstanceSegmentation` can be used for models predicting distances
            for instance segmentation, `BoundaryBasedInstanceSegmentation` for models predicting boundaries.
        image_paths: The paths to the images to use for the grid search.
        gt_paths: The paths to the labels to use for the grid search.
        result_dir: The directory for caching the grid search results.
        grid_search_values: The values to test in the grid search.
        rois: Region of interests to use for the evaluation. If given, must have the same length as `image_paths`.
        image_key: The key to the internal dataset with the image data.
            Leave None if the images are in a regular image format such as tif.
        gt_key: The key to the internal dataset with the label data.
            Leave None if the images are in a regular image format such as tif.

    Returns:
        The best parameters found by the grid search.
        The best score of the grid search.
    """
    if not HAVE_MICRO_SAM:
        raise RuntimeError(
            "The gridsearch functionality requires micro_sam. Install it via `conda install -c conda-forge micro_sam.`"
        )

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
