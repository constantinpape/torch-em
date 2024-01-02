import numpy as np

from micro_sam.instance_segmentation import InstanceSegmentationWithDecoder
from micro_sam.evaluation.instance_segmentation import (
    default_grid_search_values_instance_segmentation_with_decoder,
    evaluate_instance_segmentation_grid_search,
    run_instance_segmentation_grid_search,
)

from ..transform.raw import standardize
from .prediction import predict_with_padding, predict_with_halo


class DistanceBasedInstanceSegmentation(InstanceSegmentationWithDecoder):
    """Over-write micro_sam functionality so that it works for distance based
    segmentation with a U-net.
    """
    def __init__(self, model, preprocess=None, block_shape=None, halo=None):
        self._model = model
        self._preprocess = standardize if preprocess is None else preprocess

        assert (block_shape is None) == (halo is None)
        self._block_shape = block_shape
        self._halo = halo

        self._foreground = None
        self._center_distances = None
        self._boundary_distances = None

        self._is_initialized = False

    @property
    def is_initialized(self):
        return self._is_initialized

    def initialize(self, data):
        device = next(iter(self._model.parameters())).device

        if self._block_shape is None:
            scale_factors = self._model.init_kwargs["scale_factors"]
            min_divisible = [int(np.prod([sf[i] for sf in scale_factors])) for i in range(3)]
            input_ = self._preprocess(data)
            output = predict_with_padding(self._model, input_, min_divisible, device)
        else:
            output = predict_with_halo(
                data, self._model, [device], self._block_shape, self._halo,
                preprocess=self._preprocess,
            )

        self._foreground = output[0]
        self._center_distances = output[1]
        self._boundary_distances = output[2]

        self._is_initialized = True


def grid_search_for_distance_based_segmentation(
    segmenter, image_paths, gt_paths, result_dir,
    grid_search_values=None, rois=None,
    image_key=None, gt_key=None,
):
    if grid_search_values is None:
        grid_search_values = default_grid_search_values_instance_segmentation_with_decoder()
    run_instance_segmentation_grid_search(
        segmenter, grid_search_values, image_paths, gt_paths, result_dir,
        embedding_dir=None, verbose_gs=True,
        image_key=image_key, gt_key=gt_key, rois=rois,
    )
    best_kwargs, best_score = evaluate_instance_segmentation_grid_search(
        result_dir, list(grid_search_values.keys())
    )
    return best_kwargs, best_score
