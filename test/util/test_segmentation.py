import unittest

import numpy as np


class TestSegmentation(unittest.TestCase):
    def _two_blob_maps(self):
        # Two well-separated square objects on a 64x64 grid.
        foreground = np.zeros((64, 64), dtype="float32")
        foreground[8:24, 8:24] = 1.0
        foreground[40:56, 40:56] = 1.0

        # Boundaries enclose each object (one pixel ring around the squares).
        boundaries = np.zeros((64, 64), dtype="float32")
        for start, stop in [(8, 24), (40, 56)]:
            boundaries[start - 1:stop + 1, start - 1] = 1.0
            boundaries[start - 1:stop + 1, stop] = 1.0
            boundaries[start - 1, start - 1:stop + 1] = 1.0
            boundaries[stop, start - 1:stop + 1] = 1.0
        return foreground, boundaries

    def _count_objects(self, seg):
        return len(np.setdiff1d(np.unique(seg), [0]))

    def test_size_filter(self):
        from torch_em.util.segmentation import size_filter

        seg = np.zeros((32, 32), dtype="uint32")
        seg[2:4, 2:4] = 1   # size 4 (small)
        seg[10:20, 10:20] = 2  # size 100 (large)
        out = size_filter(seg.copy(), min_size=20)
        self.assertEqual(self._count_objects(out), 1)
        # The remaining object should be relabeled consecutively starting at 1.
        self.assertEqual(sorted(np.unique(out).tolist()), [0, 1])

    def test_connected_components_with_boundaries(self):
        from torch_em.util.segmentation import connected_components_with_boundaries

        foreground, boundaries = self._two_blob_maps()
        seg = connected_components_with_boundaries(foreground, boundaries)
        self.assertEqual(seg.shape, foreground.shape)
        self.assertEqual(seg.dtype, np.dtype("uint64"))
        self.assertEqual(self._count_objects(seg), 2)

    def test_watershed_from_components(self):
        from torch_em.util.segmentation import watershed_from_components

        foreground, boundaries = self._two_blob_maps()
        seg = watershed_from_components(boundaries, foreground, min_size=5)
        self.assertEqual(seg.shape, foreground.shape)
        self.assertEqual(self._count_objects(seg), 2)

    def test_watershed_from_maxima(self):
        from torch_em.util.segmentation import watershed_from_maxima

        foreground, boundaries = self._two_blob_maps()
        seg = watershed_from_maxima(boundaries, foreground, min_distance=5, min_size=5)
        self.assertEqual(seg.shape, foreground.shape)
        self.assertEqual(self._count_objects(seg), 2)

    def test_watershed_from_center_and_boundary_distances(self):
        from torch_em.util.segmentation import watershed_from_center_and_boundary_distances

        foreground, _ = self._two_blob_maps()

        # Low center/boundary distances at the two object centers form the seeds.
        center_distances = np.ones((64, 64), dtype="float32")
        boundary_distances = np.ones((64, 64), dtype="float32")
        for cy, cx in [(15, 15), (47, 47)]:
            center_distances[cy - 4:cy + 4, cx - 4:cx + 4] = 0.0
            boundary_distances[cy - 4:cy + 4, cx - 4:cx + 4] = 0.0

        # distance_smoothing > 0 exercises bic.filters.gaussian_smoothing.
        seg = watershed_from_center_and_boundary_distances(
            center_distances, boundary_distances, foreground, min_size=5, distance_smoothing=1.0,
        )
        self.assertEqual(seg.shape, foreground.shape)
        self.assertEqual(self._count_objects(seg), 2)


if __name__ == "__main__":
    unittest.main()
