import unittest
import numpy as np


def affs_brute_force(seg, offsets):
    assert seg.ndim == 2
    shape = seg.shape
    aff_shape = (len(offsets),) + shape
    affs = np.zeros(aff_shape, dtype="float32")
    for y in range(seg.shape[0]):
        for x in range(seg.shape[1]):
            val = seg[y, x]
            for c, off in enumerate(offsets):
                oy, ox = y + off[0], x + off[1]
                if (oy < 0 or oy >= shape[0]) or (ox < 0 or ox >= shape[1]):
                    affs[c, y, x] = 1.0
                    continue
                oval = seg[oy, ox]
                affs[c, y, x] = 0.0 if val == oval else 1.0
    return affs


def affs_brute_force_with_mask(seg, offsets, mask_bg_transition=True):
    assert seg.ndim == 2

    shape = seg.shape
    aff_shape = (len(offsets),) + shape
    affs = np.zeros(aff_shape, dtype="float32")
    mask = np.zeros(aff_shape, dtype="float32")

    for y in range(seg.shape[0]):
        for x in range(seg.shape[1]):
            val = seg[y, x]
            for c, off in enumerate(offsets):
                oy, ox = y + off[0], x + off[1]
                if (oy < 0 or oy >= shape[0]) or (ox < 0 or ox >= shape[1]):
                    affs[c, y, x] = 1.0
                    mask[c, y, x] = 0.0
                    continue

                oval = seg[oy, ox]
                n_ignore = int(val == 0) + int(oval == 0)
                if n_ignore == 2:
                    affs[c, y, x] = 1.0
                    mask[c, y, x] = 0.0
                    continue
                if n_ignore == 1 and mask_bg_transition:
                    affs[c, y, x] = 1.0
                    mask[c, y, x] = 0.0
                    continue

                affs[c, y, x] = 0.0 if val == oval else 1.0
                mask[c, y, x] = 1.0

    return affs, mask


class TestLabelTransforms(unittest.TestCase):
    def get_labels(self, with_zero):
        shape = (64, 64)
        # shape = (6, 6)
        labels = np.random.randint(1, 6, size=shape).astype("uint64")
        if with_zero:
            bg_prob = 0.25
            bg_mask = np.random.rand(*shape) < bg_prob
            labels[bg_mask] = 0
        return labels

    def test_affinities(self):
        from torch_em.transform.label import AffinityTransform
        offsets = [
            [-1, 0], [0, -1], [-3, 0], [0, -3], [4, 5], [-3, 2]
        ]
        seg = self.get_labels(with_zero=False)
        trafo = AffinityTransform(offsets)
        affs = trafo(seg)
        expected_shape = (len(offsets),) + seg.shape
        self.assertEqual(affs.shape, expected_shape)
        expected_affs = affs_brute_force(seg, offsets)
        self.assertTrue(np.allclose(affs, expected_affs))

    def test_affinities_with_mask(self):
        from torch_em.transform.label import AffinityTransform
        offsets = [
            [-1, 0], [0, -1], [-3, 0], [0, -3], [4, 5], [-3, 2]
        ]
        n_channels = len(offsets)
        seg = self.get_labels(with_zero=True)

        trafo = AffinityTransform(offsets, ignore_label=0, add_mask=True)
        affs = trafo(seg)
        expected_shape = (2 * n_channels,) + seg.shape
        self.assertEqual(affs.shape, expected_shape)
        affs, mask = affs[:n_channels], affs[n_channels:]

        expected_affs, expected_mask = affs_brute_force_with_mask(seg, offsets)
        self.assertTrue(np.allclose(affs, expected_affs))
        self.assertTrue(np.allclose(mask, expected_mask))

    def test_affinities_with_ignore_transition(self):
        from torch_em.transform.label import AffinityTransform
        offsets = [
            [-1, 0], [0, -1], [-3, 0], [0, -3], [4, 5], [-3, 2]
        ]
        n_channels = len(offsets)
        seg = self.get_labels(with_zero=True)

        trafo = AffinityTransform(offsets, ignore_label=0, add_mask=True, include_ignore_transitions=True)
        affs = trafo(seg)
        expected_shape = (2 * n_channels,) + seg.shape
        self.assertEqual(affs.shape, expected_shape)
        affs, mask = affs[:n_channels], affs[n_channels:]

        expected_affs, expected_mask = affs_brute_force_with_mask(seg, offsets, mask_bg_transition=False)

        self.assertTrue(np.allclose(affs, expected_affs))
        self.assertTrue(np.allclose(mask, expected_mask))

    def test_distance_transform(self):
        from torch_em.transform.label import DistanceTransform
        target = (np.random.rand(128, 128) > 0.95).astype("uint8")

        trafo = DistanceTransform(normalize=True, max_distance=None)
        tnew = trafo(target)
        self.assertFalse(np.allclose(tnew, 0))
        self.assertTrue((tnew >= 0).all())
        self.assertTrue((tnew <= 1).all())

        trafo = DistanceTransform(normalize=False, max_distance=5)
        tnew = trafo(target)
        self.assertFalse(np.allclose(tnew, 0))
        self.assertTrue((tnew >= 0).all())
        self.assertTrue((tnew <= 5).all())

        trafo = DistanceTransform(normalize=False, vector_distances=True)
        tnew = trafo(target)
        self.assertEqual(tnew.shape, (3,) + target.shape)
        distances, vector_distances = tnew[0], tnew[1:]
        abs_dist = np.linalg.norm(vector_distances, axis=0)
        self.assertTrue(np.allclose(distances, abs_dist))

        trafo = DistanceTransform(normalize=True, vector_distances=True)
        tnew = trafo(target)
        self.assertEqual(tnew.shape, (3,) + target.shape)
        self.assertTrue((tnew >= -1).all())
        self.assertTrue((tnew <= 1).all())

    def test_distance_transform_empty_labels(self):
        from torch_em.transform.label import DistanceTransform
        target = np.zeros((128, 128), dtype="uint8")

        trafo = DistanceTransform(invert=True, normalize=True)
        tnew = trafo(target)
        self.assertTrue(np.allclose(tnew, 0.0))

        trafo = DistanceTransform(invert=True, normalize=False)
        tnew = trafo(target)
        self.assertTrue(np.allclose(tnew, 0.0))

        trafo = DistanceTransform(invert=False, normalize=False)
        tnew = trafo(target)
        self.assertTrue(np.allclose(tnew, np.linalg.norm([128, 128])))

        trafo = DistanceTransform(invert=False, normalize=False, max_distance=10)
        tnew = trafo(target)
        self.assertTrue(np.allclose(tnew, 10.0))

        trafo = DistanceTransform(invert=False, normalize=True)
        tnew = trafo(target)
        self.assertTrue(np.allclose(tnew, 1.0))


if __name__ == "__main__":
    unittest.main()
